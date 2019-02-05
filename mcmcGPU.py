#!/usr/bin/env python
#
#Copyright 2018 Allan Haldane.

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
import scipy
import numpy as np
from numpy.random import RandomState
import pyopencl as cl
import pyopencl.array as cl_array
import os, time
import textwrap
from utils import printsome

cf = cl.mem_flags

################################################################################

# The GPU performs two main types of computation: MCMC runs, and perturbed
# coupling updates.  All MCMCGPU methods are asynchronous on the host.
# Functions that return data do not return the data directly, but return a
# FutureBuf object. The data may be obtained by FutureBuf.read(), which is
# blocking.

# The gpu has two sequence buffers: A "small" buffer for MCMC gpu generation,
# and an optional "large buffer" for combined sequence sets.

# The opencl queue is created as an out-of-order queue, and so kernel order is
# managed by the MCMCGPU class itself. By default, it makes all opencl
# commands wait until the last command is finished, but all methods also
# have a wait_for argument to override this. `None` means wait until the last
# command is done, or it can be a list of opencl events to wait for. Set it
# to the empty list [] to run immediately.

# Note that in openCL implementations there is generally a limit on the number
# of queued items allowed in a context. If you reach the limit, all queues will
# block until a kernel finishes. So we must be careful not to fill up a single
# queue before others, ie do `for i in range(100): for g in gpus: g.command()`
# instead of `for g in gpus: for i in range(100): g.command()` as the latter
# may fill the first gpu's queue, blocking the rest. 
# See  CL_DEVICE_QUEUE_ON_DEVICE_MAX_SIZE  and  CL_DEVICE_MAX_ON_DEVICE_EVENTS

# Note that on some systems there is a watchdog timer that kills any kernel
# that takes too long to finish. You will get a CL_OUT_OF_RESOURCES error
# if this happens, which occurs when the *following* kernel is run.

class FutureBuf:
    def __init__(self, buffer, event, postprocess=None):
        self.buffer = buffer
        self.event = event
        self.postfunc = postprocess

    def read(self):
        self.event.wait()
        if self.postfunc != None:
            return self.postfunc(self.buffer)
        return self.buffer

def readGPUbufs(bufnames, gpus):
    futures = [[gpu.getBuf(bn) for gpu in gpus] for bn in bufnames]
    return [[buf.read() for buf in gpuf] for gpuf in futures]

class MCMCGPU:
    def __init__(self, gpuinfo, L, q, nseq, wgsize, outdir,
                 vsize, seed, profile=False):
        if nseq%512 != 0:
            raise ValueError("nwalkers/ngpus must be a multiple of 512")
            # this guarantees that all kernel access to seqmem is coalesced and
            # simplifies the histogram kernels
        
        self.L = L
        self.q = q
        self.nPairs = L*(L-1)//2
        self.events = []
        self.SWORDS = ((L-1)//4+1)    #num words needed to store a sequence
        self.SBYTES = (4*self.SWORDS) #num bytes needed to store a sequence
        self.nseq = {'main': nseq}

        gpu, gpunum, ctx, prg = gpuinfo
        self.gpunum = gpunum
        self.ctx = ctx
        self.prg = prg

        self.wgsize = wgsize
        self.nhist, self.histws = histogram_heuristic(q)

        self.logfn = os.path.join(outdir, 'gpu-{}.log'.format(gpunum))
        with open(self.logfn, "wt") as f:
            printDevice(f.write, gpu)

        self.mcmcprg = prg.metropolis

        self.rngstate = RandomState(seed)

        #setup opencl for this device
        self.log("Getting CL Queue")

        qprop = cl.command_queue_properties.OUT_OF_ORDER_EXEC_MODE_ENABLE
        self.profile = profile
        if profile:
            qprop |= cl.command_queue_properties.PROFILING_ENABLE
        self.queue = cl.CommandQueue(ctx, device=gpu, properties=qprop)

        self.log("\nOpenCL Device Compilation Log:")
        self.log(self.prg.get_build_info(gpu, cl.program_build_info.LOG))
        maxwgs = self.mcmcprg.get_work_group_info(
                 cl.kernel_work_group_info.WORK_GROUP_SIZE, gpu)
        self.log("Max MCMC WGSIZE: {}".format(maxwgs))

        self.initted = []

        self.bufs = {}
        self.buf_spec = {}
        self.seqbufs = {}
        self.Ebufs = {}
        self.largebufs = []

        # setup essential buffers
        # note no padding needed on seq buffers since alignment is guaranteed
        nPairs, SWORDS = self.nPairs, self.SWORDS
        self._setupBuffer('Junpacked', '<f4', (L*L, q*q))
        self._setupBuffer(        'J', '<f4', (nPairs, q*q))
        self._setupBuffer(       'bi', '<f4', (nPairs, q*q)),
        self._setupBuffer(  'bicount', '<u4', (nPairs, q*q)),
        self._setupBuffer( 'seq main', '<u4', (SWORDS, self.nseq['main'])),
        self._setupBuffer('seqL main', '<u4', (L, self.nseq['main']//4)),
        self._setupBuffer(   'E main', '<f4', (self.nseq['main'],)),
        self.unpackedJ = False #use to keep track of whether J is unpacked
        self.repackedSeqT = {'main': False}

        self.lastevt = None

    def log(self, str):
        #logs are rare, so just open the file every time
        with open(self.logfn, "at") as f:
            print(repr(time.process_time()), str, file=f)

    def logevt(self, name, evt, nbytes=None):
        self.lastevt = evt

        # don't save events if not profiling.
        # note that saved events use up memory - free it using logprofile
        if self.profile:
            if len(self.events)%1000 == 0 and self.events != []:
                self.log("Warning: Over {} profiling events are not flushed "
                    "(using up memory)".format(len(self.events)))
            if nbytes:
                self.events.append((evt, name, nbytes))
            else:
                self.events.append((evt, name))

        return evt

    def _waitevt(self, evts=None):
        if evts is not None:
            if isinstance(evts, cl.Event):
                return [evts]
            return evts
        if self.lastevt is not None:
            return [self.lastevt]

    def logProfile(self):
        self.wait()

        if not self.profile:
            return
        with open(self.logfn, "at") as f:
            while self.events != []:
                dat = self.events.pop(0)
                evt, name, size = dat[0],dat[1],(dat[2] if len(dat)==3 else '')
                print("EVT", name, evt.profile.start, evt.profile.end,
                      size, file=f)

    def _setupBuffer(self, bufname, buftype, bufshape, flags=cf.READ_WRITE):
        flags = flags | cf.ALLOC_HOST_PTR
        size = np.dtype(buftype).itemsize*int(np.product(bufshape))
        buf = cl.Buffer(self.ctx, flags, size=size)

        self.bufs[bufname] = buf
        self.buf_spec[bufname] = (buftype, bufshape, flags)

        # add it to convenience dicts if applicable
        names = bufname.split()
        if len(names) > 1:
            bufs = {'seq': self.seqbufs, 'E': self.Ebufs}
            if names[0] in bufs:
                bufs[names[0]][names[1]] = buf

    def require(self, *reqs):
        for r in reqs:
            if r not in self.initted:
                raise Exception("{} not initialized".format(r))

    def _initcomponent(self, cmp):
        if cmp in self.initted:
            raise Exception("Already initialized {}".format(cmp))
        self.initted.append(cmp)

    def initMCMC(self, nsteps, nMCMCcalls):
        self._initcomponent('MCMC')

        # rngstates should be size of mwc64xvec2_state_t
        self.nsteps = nsteps
        self._setupBuffer('rngstates', '<2u8', (self.nseq['main'],)),
        self._setupBuffer(       'Bs', '<f4',  (self.nseq['main'],)),
        self._setupBuffer(  'randpos', '<u4',  (self.nsteps,))

        self.setBuf('Bs', np.ones(self.nseq['main'], dtype='<f4'))
        self._initMCMC_RNG(nMCMCcalls)
        self.nsteps = int(nsteps)

    def initLargeBufs(self, nseq_large):
        self._initcomponent('Large')

        self.nseq['large'] = nseq_large
        self._setupBuffer(    'seq large', '<u4', (self.SWORDS, nseq_large))
        self._setupBuffer(   'seqL large', '<u4', (self.L, nseq_large//4)),
        self._setupBuffer(      'E large', '<f4', (nseq_large,))
        self._setupBuffer('weights large', '<f4',  (nseq_large,))

        self.largebufs.extend(['seq large', 'seqL large', 'E large',
                               'weights large'])
        self.nstoredseqs = 0

        # it is important to zero out the large seq buffer, because
        # if it is partially full we may need to compute energy
        # over the padding sequences at the end to get a full wg.
        buf = self.bufs['seq large']
        self.fillBuf('seq large', 0)
        self.repackedSeqT['large'] = False

    def initSubseq(self):
        self.require('Large')
        self._initcomponent('Subseq')
        self._setupBuffer('markpos', '<u1',  (self.SBYTES,), cf.READ_ONLY)
        self.markPos(np.zeros(self.SBYTES, '<u1'))

    # we may want to select replicas at a particular temperature
    def initMarkSeq(self):
        self._initcomponent('Markseq')

        self._setupBuffer( 'markseqs', '<i4',  (self.nseq['main'],))
        self.setBuf('markseqs', np.arange(self.nseq['main'], dtype='<i4'))
        self.nmarks = self.nseq['main']

    def initJstep(self):
        self._initcomponent('Jstep')

        nPairs, q = self.nPairs, self.q
        self._setupBuffer('bi target', '<f4',  (nPairs, q*q))
        self._setupBuffer(     'Creg', '<f4',  (nPairs, q*q))
        self._setupBuffer( 'Xlambdas', '<f4',  (nPairs,))
        self._setupBuffer(     'neff', '<f4',  (1,))
        self._setupBuffer(  'weights', '<f4',  (self.nseq['main'],))


    def packSeqs_4(self, seqs):
        """
        Converts seqs to 4-byte uint format on CPU, padded to 32bits, assumes
        little endian. Each row's bytes are 
            a0 a1 a2 a3 b0 b1 b2 b3 c0 c1 c2 c3 ...
        for sequences a, b, c, so each uint32 correaponds to 4 seq bytes.
        """
        bseqs = np.zeros((seqs.shape[0], self.SBYTES), dtype='<u1', order='C')
        bseqs[:,:self.L] = seqs
        mem = np.zeros((self.SWORDS, seqs.shape[0]), dtype='<u4', order='C')
        for i in range(self.SWORDS):
            mem[i,:] = bseqs.view(np.uint32)[:,i]
        return mem

    def unpackSeqs_4(self, mem):
        """ reverses packSeqs_4 (on CPU)"""
        bseqs = np.zeros((mem.shape[1], self.SBYTES), dtype='<u1', order='C')
        for i in range(self.SWORDS): #undo memory rearrangement
            bseqs.view(np.uint32)[:,i] = mem[i,:]
        return bseqs[:,:self.L]

    def repackseqs_T(self, bufname, wait_for=None):
        """
        On GPU, copies the seq buffer (in 4-byte format) to a seqL buffer
        in "transpose" format, which is just the usual CPU sequence buffer 
        but transposed.
        """
        self.log("repackseqs_T")
        
        nseq = self.nseq[bufname]
        inseq_dev = self.bufs['seq ' + bufname]
        outseq_dev = self.bufs['seqL ' + bufname]

        self.repackedSeqT[bufname] = True
        return self.logevt('repackseqs_T',
            self.prg.unpackseqs1(self.queue, (self.SWORDS*256,), (256,),
                            inseq_dev, np.uint32(nseq),
                            outseq_dev, np.uint32(nseq//4),
                            wait_for=self._waitevt(wait_for)))

    def unpackJ(self, wait_for=None):
        """convert from format where every row is a unique ij pair (L choose 2
        rows) to format with every pair, all orders (L^2 rows)."""

        # quit if J already loaded/unpacked
        if self.unpackedJ:
            return wait_for

        self.log("unpackJ")

        q, nPairs = self.q, self.nPairs
        self.unpackedJ = True
        return self.logevt('unpackJ',
            self.prg.unpackfV(self.queue, (nPairs*q*q,), (q*q,),
                            self.bufs['J'], self.bufs['Junpacked'],
                            wait_for=self._waitevt(wait_for)))

    def _initMCMC_RNG(self, nMCMCcalls, wait_for=None):
        self.require('MCMC')
        self.log("initMCMC_RNG")

        rsize = 2

        nsamples = np.uint64(2**40) #upper bound for # of rngs generated
        nseq = self.nseq['main']
        offset = np.uint64(nsamples*nseq*self.gpunum*rsize)
        # each gpu uses perStreamOffset*get_global_id(0)*vectorSize samples
        #                    (nsamples *      nseq      * vecsize)

        # read mwc64 docs for description of nsamples.
        # Num rng samples should be chosen such that 2**64/(rsize*nsamples) is
        # greater than # walkers. nsamples should be > #MC steps performed
        # per walker (which is nsteps*nMCMCcalls)
        if not (self.nsteps*nMCMCcalls < nsamples < 2**64//(rsize*self.wgsize)):
            raise Exception("RNG problem. RNGs may not be independent.")
        #if this is a problem rethink the value 2**40 above, or consider using
        #an rng with a greater period, eg the "Warp" generator.

        wgsize = self.wgsize
        while wgsize > nseq:
            wgsize = wgsize//2
        return self.logevt('initMCMC_RNG',
            self.prg.initRNG2(self.queue, (nseq,), (wgsize,),
                         self.bufs['rngstates'], offset, nsamples,
                         wait_for=self._waitevt(wait_for)))

    def runMCMC(self, wait_for=None):
        """Performs a single round of mcmc sampling (nsteps MC steps)"""
        self.require('MCMC')
        self.log("runMCMC")

        nseq = self.nseq['main']
        nsteps = self.nsteps
        wait_for = self.unpackJ(wait_for=self._waitevt(wait_for))

        # all gpus use same rng series. This way there is no difference
        # between running on one gpu vs splitting on multiple
        rng = self.rngstate.randint(0, self.L, size=nsteps).astype('u4')
        self.setBuf('randpos', rng)

        self.repackedSeqT['main'] = False
        return self.logevt('mcmc',
            self.mcmcprg(self.queue, (nseq,), (self.wgsize,),
                         self.bufs['Junpacked'], self.bufs['rngstates'],
                         self.bufs['randpos'], np.uint32(nsteps),
                         self.Ebufs['main'], self.bufs['Bs'],
                         self.seqbufs['main'],
                         wait_for=self._waitevt(wait_for)))


    def measureFPerror(self, log, nloops=3):
        log("Measuring FP Error")
        for n in range(nloops):
            self.runMCMC()
            e1 = self.getBuf('E main').read()
            self.calcEnergies('main')
            e2 = self.getBuf('E main').read()
            log("Run", n, "Error:", np.mean((e1-e2)**2))
            log('    Final E MC', printsome(e1), '...')
            log("    Final E rc", printsome(e2), '...')

            seqs = self.getBuf('seq main').read()
            J = self.getBuf('J').read()
            e3 = getEnergies(seqs, J)
            log("    Exact E", e3[:5])
            log("    Error:", np.mean([float((a-b)**2) for a,b in zip(e1, e3)]))

    def calcBicounts(self, seqbufname, wait_for=None):
        self.log("calcBicounts " + seqbufname)
        L, q, nPairs, nhist = self.L, self.q, self.nPairs, self.nhist

        if seqbufname == 'main':
            nseq = self.nseq[seqbufname]
            buflen = nseq
        else:
            nseq = self.nstoredseqs
            buflen = self.nseq[seqbufname]
        seq_dev = self.seqbufs[seqbufname]

        localhist = cl.LocalMemory(nhist*q*q*np.dtype(np.uint32).itemsize)
        return self.logevt('calcBicounts',
            self.prg.countBivariate(self.queue, (nPairs*nhist,), (nhist,),
                     self.bufs['bicount'],
                     np.uint32(nseq), seq_dev, np.uint32(buflen), localhist,
                     wait_for=self._waitevt(wait_for)))

    def bicounts_to_bimarg(self, nseq, wait_for=None):
        self.log("bicounts_to_bimarg ")
        q, nPairs = self.q, self.nPairs
        nworkunits = self.wgsize*((nPairs*q*q-1)//self.wgsize+1)
        return self.logevt('bicounts_to_bimarg',
            self.prg.bicounts_to_bimarg(self.queue,
                     (nworkunits,), (self.wgsize,),
                     self.bufs['bicount'], self.bufs['bi'], np.uint32(nseq),
                     wait_for=self._waitevt(wait_for)))

    def calcEnergies(self, seqbufname, wait_for=None):
        self.log("calcEnergies " + seqbufname)

        energies_dev = self.Ebufs[seqbufname]
        seq_dev = self.seqbufs[seqbufname]
        buflen = self.nseq[seqbufname]

        if seqbufname == 'main':
            nseq = self.nseq[seqbufname]
        else:
            nseq = self.nstoredseqs
            # pad to be a multiple of wgsize (uses dummy seqs at end)
            nseq = nseq + ((self.wgsize - nseq) % self.wgsize)

        return self.logevt('getEnergies',
            self.prg.getEnergies(self.queue, (nseq,), (self.wgsize,),
                             self.bufs['J'], seq_dev, np.uint32(buflen),
                             energies_dev, wait_for=self._waitevt(wait_for)))

    def calcWeights(self, seqbufname='main', wait_for=None):
        #overwrites weights, neff
        #assumes seqmem_dev, energies_dev are filled in
        self.require('Jstep')
        self.log("calcWeights")

        if seqbufname == 'main':
            nseq = self.nseq[seqbufname]
            buflen = nseq
            weights_dev = self.bufs['weights']
        else:
            nseq = self.nstoredseqs
            # pad to be a multiple of wgsize (uses dummy seqs at end)
            nseq = nseq + ((self.wgsize - nseq) % self.wgsize)

            buflen = self.nseq[seqbufname]
            weights_dev = self.bufs['weights large']
        seq_dev = self.seqbufs[seqbufname]
        E_dev = self.Ebufs[seqbufname]

        return self.logevt('perturbedWeights',
            self.prg.perturbedWeights(self.queue, (nseq,), (self.wgsize,),
                           self.bufs['J'], seq_dev, np.uint32(buflen),
                           weights_dev, E_dev,
                           wait_for=self._waitevt(wait_for)))
    
    def weightedMarg(self, seqbufname='main', wait_for=None):
        self.require('Jstep')
        self.log("weightedMarg")
        q, L, nPairs = self.q, self.L, self.nPairs
        nhist, histws = self.nhist, self.histws

        if seqbufname == 'main':
            nseq = self.nseq[seqbufname]
            buflen = nseq//4
            weights_dev = self.bufs['weights']
        else:
            nseq = self.nstoredseqs
            buflen = self.nseq[seqbufname]//4
            weights_dev = self.bufs['weights large']
            # pad to be a multiple of wgsize (uses dummy seqs at end)
            nseq = nseq + ((self.wgsize - nseq) % self.wgsize)

        if not self.repackedSeqT[seqbufname]:
            wait_for = self.repackseqs_T(seqbufname,
                                      wait_for=self._waitevt(wait_for))
        seq_dev = self.bufs['seqL ' + seqbufname]

        return self.logevt('weightedMarg',
            self.prg.weightedMarg(self.queue, (nPairs*histws,), (histws,),
                        self.bufs['bi'], weights_dev,
                        np.uint32(nseq), seq_dev, np.uint32(buflen),
                        wait_for=self._waitevt(wait_for)))

    def renormalize_bimarg(self, wait_for=None):
        self.log("renormalize_bimarg")
        q, nPairs = self.q, self.nPairs
        return self.logevt('renormalize_bimarg',
            self.prg.renormalize_bimarg(self.queue, (nPairs*q*q,), (q*q,),
                         self.bufs['bi'], wait_for=self._waitevt(wait_for)))

    def addBiBuffer(self, bufname, otherbuf, wait_for=None):
        # used for combining results from different gpus, where  otherbuf is a
        # buffer "belonging" to another gpu
        self.log("addbuf")

        selfbuf = self.bufs[bufname]
        if selfbuf.size != otherbuf.size:
            raise Exception('Tried to add bufs of different sizes')

        q, nPairs = self.q, self.nPairs
        nworkunits = self.wgsize*((nPairs*q*q-1)//self.wgsize+1)

        return self.logevt('addbuf',
            self.prg.addBiBufs(self.queue, (nworkunits,), (self.wgsize,),
                       selfbuf, otherbuf, wait_for=self._waitevt(wait_for)))

    def updateJ(self, gamma, pc, wait_for=None):
        self.require('Jstep')
        self.log("updateJ")
        q, nPairs = self.q, self.nPairs
        #find next highest multiple of wgsize, for num work units
        nworkunits = self.wgsize*((nPairs*q*q-1)//self.wgsize+1)

        bibuf = self.bufs['bi']
        Jin = Jout = self.bufs['J']
        self.unpackedJ = False
        return self.logevt('updateJ',
            self.prg.updatedJ(self.queue, (nworkunits,), (self.wgsize,),
                                self.bufs['bi target'], bibuf,
                                np.float32(gamma), np.float32(pc), Jin, Jout,
                                wait_for=self._waitevt(wait_for)))

    def updateJ_l2z(self, gamma, pc, lh, lJ, wait_for=None):
        self.require('Jstep')
        self.log("updateJ_l2z")
        q, nPairs = self.q, self.nPairs

        bibuf = self.bufs['bi']
        Jin = Jout = self.bufs['J']
        self.unpackedJ = None
        return self.logevt('updateJ_l2z',
            self.prg.updatedJ_l2z(self.queue, (nPairs*q*q,), (q*q,),
                            self.bufs['bi target'], bibuf,
                            np.float32(gamma), np.float32(pc),
                            np.float32(2*lh), np.float32(2*lJ), Jin, Jout,
                            wait_for=self._waitevt(wait_for)))

    def updateJ_X(self, gamma, pc, wait_for=None):
        self.require('Jstep')
        self.log("updateJ X")
        q, nPairs = self.q, self.nPairs

        bibuf = self.bufs['bi']
        Jin = Jout = self.bufs['J']
        self.unpackedJ = None
        return self.logevt('updateJ_X',
            self.prg.updatedJ_X(self.queue, (nPairs*q*q,), (q*q,),
                                self.bufs['bi target'], bibuf,
                                self.bufs['Creg'],
                                np.float32(gamma), np.float32(pc), Jin, Jout,
                                wait_for=self._waitevt(wait_for)))

    def updateJ_Xself(self, gamma, pc, wait_for=None):
        self.require('Jstep')
        self.log("updateJ Xself")
        q, nPairs = self.q, self.nPairs

        bibuf = self.bufs['bi']
        Jin = Jout = self.bufs['J']
        self.unpackedJ = None
        return self.logevt('updateJ_Xself',
            self.prg.updatedJ_Xself(self.queue, (nPairs*q*q,), (q*q,),
                                self.bufs['bi target'], bibuf,
                                self.bufs['Xlambdas'],
                                np.float32(gamma), np.float32(pc), Jin, Jout,
                                wait_for=self._waitevt(wait_for)))

    def getBuf(self, bufname, truncateLarge=True, wait_for=None):
        """get buffer data. truncateLarge means only return the
        computed part of the large buffer (rest may be uninitialized)"""

        self.log("getBuf " + bufname)
        bufspec = self.buf_spec[bufname]
        buftype, bufshape = bufspec[0], bufspec[1]
        mem = np.zeros(bufshape, dtype=buftype)
        evt = cl.enqueue_copy(self.queue, mem, self.bufs[bufname],
                          is_blocking=False, wait_for=self._waitevt(wait_for))
        self.logevt('getBuf', evt, mem.nbytes)
        if bufname.split()[0] == 'seq':
            if bufname in self.largebufs and truncateLarge:
                nret = self.nstoredseqs
                return FutureBuf(mem, evt,
                                 lambda b: self.unpackSeqs_4(b)[:nret,:])
            return FutureBuf(mem, evt, self.unpackSeqs_4)

        if bufname in self.largebufs and truncateLarge:
            nret = self.nstoredseqs
            return FutureBuf(mem, evt, lambda b: b[:nret])

        return FutureBuf(mem, evt)

    def setBuf(self, bufname, buf, wait_for=None):
        self.log("setBuf " + bufname)

        # device-to-device copies skip all the checks
        if isinstance(buf, cl.Buffer):
            evt = cl.enqueue_copy(self.queue, self.bufs[bufname], buf,
                                  wait_for=self._waitevt(wait_for))
            self.logevt('setBuf', evt, buf.size)
            if bufname.split()[0] == 'J':
                self.unpackedJ = None
            return  evt

        if bufname.split()[0] == 'seq':
            buf = self.packSeqs_4(buf)

        bufspec = self.buf_spec[bufname]
        buftype, bufshape = bufspec[0], bufspec[1]
        if not isinstance(buf, np.ndarray):
            buf = array(buf, dtype=buftype)

        if np.dtype(buftype) != buf.dtype:
            raise ValueError("Buffer dtype mismatch.Expected {}, got {}".format(
                             np.dtype(buftype), buf.dtype))
        if bufshape != buf.shape and not (bufshape == (1,) or buf.size == 1):
            raise ValueError("Buffer size mismatch. Expected {}, got {}".format(
                            bufshape, buf.shape))

        evt = cl.enqueue_copy(self.queue, self.bufs[bufname], buf,
                            is_blocking=False, wait_for=self._waitevt(wait_for))
        self.logevt('setBuf', evt, buf.nbytes)
        #unset packedJ flag if we modified that J buf
        if bufname.split()[0] == 'J':
            self.unpackedJ = None
        if bufname == 'seq large':
            self.nstoredseqs = bufshape[1]
        if bufname.split()[0] == 'seq':
            self.repackedSeqT[bufname.split()[1]] = False

        return evt

    def fillBuf(self, bufname, val, wait_for=None):
        self.log("fillBuf " + bufname)

        buf = self.bufs[bufname]
        buftype = np.dtype(self.buf_spec[bufname][0]).type

        self.logevt('fill_buffer',
            cl.enqueue_fill_buffer(self.queue, buf, buftype(val), 0, buf.size,
                                   wait_for=self._waitevt()))

    def markPos(self, marks, wait_for=None):
        self.require('Subseq')

        marks = marks.astype('<u1')
        if len(marks) == self.L:
            marks.resize(self.SBYTES)
        return self.setBuf('markpos', marks, wait_for=wait_for)

    def fillSeqs(self, startseq, seqbufname='main', wait_for=None):
        #write a kernel function for this?
        self.log("fillSeqs " + seqbufname)
        nseq = self.nseq[seqbufname]
        self.setBuf('seq '+seqbufname, np.tile(startseq, (nseq,1)),
                    wait_for=wait_for)

    def storeSeqs(self, seqs=None, wait_for=None):
        """
        If seqs is None, stores main to large seq buffer. Otherwise
        stores seqs to large buffer
        """

        self.require('Large')

        offset = self.nstoredseqs
        self.log("storeSeqs " + str(offset))

        if seqs is not None:
            nseq, L = seqs.shape
            if L != self.L:
                raise Exception(
                    "Sequences have wrong length: {} vs {}".format(L, self.L))
            if offset + nseq > self.nseq['large']:
                raise Exception("cannot store seqs past end of large buffer")
            assert(seqs.dtype == np.dtype('u1'))
            buf = self.packSeqs_4(seqs)

            w, h = self.buf_spec['seq large'][1] # L/4, nseq
            # for some reason, rectangular copies in pyOpencl use opposite axis
            # order from numpy, and need indices in bytes not elements, so we
            # have to switch all this around. buf is uint32, or 4 bytes.
            evt = cl.enqueue_copy(self.queue, self.seqbufs['large'], buf,
                                  buffer_origin=(4*offset, 0),
                                  host_origin=(0, 0),
                                  region=(4*buf.shape[1], buf.shape[0]),
                                  buffer_pitches=(4*h, w),
                                  host_pitches=(4*buf.shape[1], buf.shape[0]),
                                  is_blocking=False,
                                  wait_for=self._waitevt(wait_for))
        else:
            nseq = self.nseq['main']
            if offset + nseq > self.nseq['large']:
                raise Exception("cannot store seqs past end of large buffer")
            evt = self.prg.storeSeqs(self.queue, (nseq,), (self.wgsize,),
                               self.seqbufs['main'], self.seqbufs['large'],
                               np.uint32(self.nseq['large']), np.uint32(offset),
                               wait_for=self._waitevt(wait_for))

        self.nstoredseqs += nseq
        self.repackedSeqT['large'] = False
        return self.logevt('storeSeqs', evt)

    def markSeqs(self, mask, wait_for=None):
        self.require('Markseq')

        marks = -np.ones(mask.shape, dtype='i4')
        inds = np.where(mask)[0]
        marks[inds] = np.arange(len(inds), dtype='i4')
        self.setBuf('markseqs', marks, wait_for=wait_for)
        self.nmarks = len(inds)
        self.log("marked {} seqs".format(len(inds)))

    def storeMarkedSeqs(self, wait_for=None):
        self.require('Markseq', 'Large')

        nseq = self.nseq['main']
        newseq = self.nmarks
        if nseq == newseq:
            self.storeSeqs()
            return

        offset = self.nstoredseqs
        self.log("storeMarkedSeqs {} {}".format(offset, newseq))

        if offset + newseq > self.nseq['large']:
            raise Exception("cannot store seqs past end of large buffer")
        self.nstoredseqs += newseq
        self.repackedSeqT['large'] = False
        return self.logevt('storeMarkedSeqs',
            self.prg.storeMarkedSeqs(self.queue, (nseq,), (self.wgsize,),
                           self.seqbufs['main'], self.seqbufs['large'],
                           np.uint32(self.nseq['large']), np.uint32(offset),
                           self.bufs['markseqs'],
                           wait_for=self._waitevt(wait_for)))

    def clearLargeSeqs(self):
        self.require('Large')
        self.nstoredseqs = 0
        self.repackedSeqT['large'] = False

    def restoreSeqs(self, wait_for=None):
        self.require('Large')
        self.log("restoreSeqs " + str(offset))
        nseq = self.nseq['main']

        if offset + nseq > self.nseq['large']:
            raise Exception("cannot get seqs past end of large buffer")
        if self.nstoredseqs < nseq:
            raise Exception("not enough seqs stored in large buffer")

        self.repackedSeqT['main'] = False
        return self.logevt('restoreSeqs',
            self.prg.restoreSeqs(self.queue, (nseq,), (self.wgsize,),
                           self.seqbufs['main'], self.seqbufs['large'],
                           np.uint32(self.nseq['large']), np.uint32(offset),
                           wait_for=self._waitevt(wait_for)))

    def copySubseq(self, seqind, wait_for=None):
        self.require('Subseq')
        self.log("copySubseq " + str(seqind))
        nseq = self.nseq['large']
        if seqind >= self.nseq['main']:
            raise Exception("given index is past end of main seq buffer")
        self.repackedSeqT['large'] = False
        return self.logevt('copySubseq',
            self.prg.copySubseq(self.queue, (nseq,), (self.wgsize,),
                            self.seqbufs['main'], self.seqbufs['large'],
                            np.uint32(self.nseq['main']), np.uint32(seqind),
                            self.bufs['markpos'],
                            wait_for=self._waitevt(wait_for)))

    def wait(self):
        self.log("wait")
        self.queue.finish()

################################################################################
# Set up enviroment and some helper functions

os.environ['PYOPENCL_COMPILER_OUTPUT'] = '0'
os.environ['PYOPENCL_NO_CACHE'] = '1'
os.environ["CUDA_CACHE_DISABLE"] = '1'

def printPlatform(log, p, n=0):
    log("Platform {} '{}':".format(n, p.name))
    log("    Vendor: {}".format(p.vendor))
    log("    Version: {}".format(p.version))
    exts = ("\n" + " "*16).join(textwrap.wrap(p.extensions, 80-16))
    log("    Extensions: {}".format(exts))

def printDevice(log, d):
    log("  Device '{}':".format(d.name))
    log("    Vendor: {}".format(d.vendor))
    log("    Version: {}".format(d.version))
    log("    Driver Version: {}".format(d.driver_version))
    log("    Max Clock Frequency: {}".format(d.max_clock_frequency))
    log("    Max Compute Units: {}".format(d.max_compute_units))
    log("    Max Work Group Size: {}".format(d.max_work_group_size))
    log("    Global Mem Size: {}".format(d.global_mem_size))
    log("    Global Mem Cache Size: {}".format(d.global_mem_cache_size))
    log("    Local Mem Size: {}".format(d.local_mem_size))
    log("    Max Constant Buffer Size: {}".format(d.max_constant_buffer_size))

def printGPUs(log):
    for n,p in enumerate(cl.get_platforms()):
        printPlatform(log, p, n)
        for d in p.get_devices():
            printDevice(log, d)
        log("")

def unpackJ_CPU(self, couplings):
    """convert from format where every row is a unique ij pair (L choose 2
    rows) to format with every pair, all orders (L^2 rows). Note that the
    GPU kernel unpackfV does the same thing faster"""

    L, q = seqsize_from_param_shape(couplings.shape)
    fullcouplings = np.zeros((L*L,q*q), dtype='<f4', order='C')
    pairs = [(i,j) for i in range(L-1) for j in range(i+1,L)]
    for n,(i,j) in enumerate(pairs):
        c = couplings[n,:]
        fullcouplings[L*i + j,:] = c
        fullcouplings[L*j + i,:] = c.reshape((q,q)).T.flatten()
    return fullcouplings

################################################################################

def setupGPUs(scriptpath, scriptfile, param, log):
    outdir = param.outdir
    L, q = param.L, param.q
    gpuspec = param.gpuspec
    measureFPerror = param.fperror

    with open(scriptfile) as f:
        src = f.read()

    #figure out which gpus to use
    gpudevices = []
    platforms = [(p, p.get_devices()) for p in cl.get_platforms()]
    if gpuspec is not None:
        try:
            dev = [tuple(int(x) for x in a.split(':'))
                                for a in gpuspec.split(',')]
        except:
            raise Exception("Error: GPU specification must be comma separated "
                            " list of platforms, eg '0,0'")

        for p,d in dev:
            try:
                plat, devices = platforms[p]
                gpu = devices.popitem()[d]
            except:
                raise Exception("No GPU with specification {}".format(d))
            log("Using GPU {} on platform {} ({})".format(gpu.name,
                                                          plat.name, d))
            gpudevices.append(gpu)
    else:
        #use gpus in first platform
        plat = platforms[0]
        for gpu in plat[1]:
            log("Using GPU {} on platform {} (0)".format(gpu.name,
                                                         plat[0].name))
            gpudevices.append(gpu)

    if len(gpudevices) == 0:
        raise Exception("Error: No GPUs found")

    #set up OpenCL. Assumes all gpus are identical
    log("Getting CL Context...")
    cl_ctx = cl.Context(gpudevices)

    nhist, histws = histogram_heuristic(q)

    #compile CL program
    options = [('q', q), ('L', L), ('NHIST', nhist), ('HISTWS', histws)]
    if measureFPerror:
        options.append(('MEASURE_FP_ERROR', 1))
    optstr = " ".join(["-D {}={}".format(opt,val) for opt,val in options])
    log("Compilation Options: ", optstr)
    extraopt = " -cl-nv-verbose -Werror -I {}".format(scriptpath)
    log("Compiling CL...")
    cl_prg = cl.Program(cl_ctx, src).build(optstr + extraopt)

    #dump compiled program
    ptx = cl_prg.get_info(cl.program_info.BINARIES)
    for n,p in enumerate(ptx):
        #useful to see if compilation changed
        #log("PTX length: ", len(p))
        with open(os.path.join(outdir, 'ptx{}'.format(n)), 'wt') as f:
            f.write(p.decode('utf-8'))

    return (cl_ctx, cl_prg), gpudevices

def divideWalkers(nwalkers, ngpus, wgsize, log):
    n_max = (nwalkers-1)//ngpus + 1
    nwalkers_gpu = [n_max]*(ngpus-1) + [nwalkers - (ngpus-1)*n_max]
    if nwalkers % (ngpus*wgsize) != 0:
        log("Warning: number of MCMC walkers is not a multiple of "
            "wgsize*ngpus, so there are idle work units.")
    return nwalkers_gpu

def initGPU(devnum, cldat, device, nwalkers, param, log):
    cl_ctx, cl_prg = cldat
    outdir = param.outdir
    L, q = param.L, param.q
    profile = param.profile
    wgsize = param.wgsize
    seed = param.rngseed

    # wgsize = OpenCL work group size for MCMC kernel.
    # (also for other kernels, although would be nice to uncouple them)
    if wgsize not in [1<<n for n in range(32)]:
        raise Exception("wgsize must be a power of two")

    vsize = 1024 #power of 2. Work group size for 1d vector operations.

    log("Starting GPU {}".format(devnum))
    gpu = MCMCGPU((device, devnum, cl_ctx, cl_prg), L, q,
                  nwalkers, wgsize, outdir, vsize, seed, profile=profile)
    return gpu

def histogram_heuristic(q):
    """
    Choose histogram size parameters (NHIST, HISTWS) for the bimarg GPU
    calculations. 
    
    Each histogram is q*q float32, and we want to squeeze as many as possible
    into local memory (eg, 96k), such that multiple workgroups can run at
    once. Strategy below figues out # of histograms which can fit into 16k.
    Then we figure out the optimal wg size histws that does not waste too many
    work units, since in the worst part of the kernel only nhist wu are
    running. We also want nhist, histws to be powers of 2, and histws > nhist.
    """
    nhist = 4096//(q*q)
    if nhist == 0:
        raise Exception("alphabet size too large to make histogram on gpu")
    nhist = 2**int(np.log2(nhist)) # closest power of two
    
    # this seems like a roughly good heuristic on Titan X.
    if q <= 12:
        hist_ws = 512
    elif q <= 16:
        hist_ws = 256
    else:
        hist_ws = 128

    if nhist > hist_ws:
        nhist = hist_ws

    return nhist, hist_ws

def merge_device_bimarg(gpus):
    # each gpu has its own bimarg computed for its sequences. We want to sum
    # the bimarg to get the total bimarg, so need to share the bimarg buffers
    # across devices. Since gpus are in same context, they can share buffers.

    # There are a few possible transfer strategies with different bus
    # usage/sync issues. The one below uses a divide by two strategy: In the
    # first round, second half of gpus send to first half, eg gpu 3 to 1, 2 to
    # 0, which do a sum. Then repeat on first half of gpus, dividing gpus in
    # half each round, then broadcast final result from gpu 0.

    # make sure we are done writing to buffer
    for g in gpus:
        g.wait()

    rgpus = gpus
    while len(rgpus) > 1:
        h = (len(rgpus)-1)//2 + 1
        for even, odd in zip(rgpus[:h], rgpus[h:]):
            even.addBiBuffer('bi', odd.bufs['bi'])

        for even in rgpus[:h]:
            even.wait()

        rgpus = rgpus[:h]

    rgpus[0].renormalize_bimarg()
    rgpus[0].wait()

    for g in gpus[1:]:
        g.setBuf('bi', rgpus[0].bufs['bi'])

    for g in gpus[1:]:
        g.wait()


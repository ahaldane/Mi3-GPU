#!/usr/bin/env python2
from __future__ import print_function
from scipy import *
import scipy
import numpy as np
from numpy.random import randint
import numpy.random
import pyopencl as cl
import pyopencl.array as cl_array
import os, time
import seqload
import textwrap

cf = cl.mem_flags

################################################################################

#The GPU performs two types of computation: MCMC runs, and perturbed
#coupling updates.  All MCMCGPU methods are asynchronous. Functions that return
#data do not return the data directly, but return a FutureBuf object. The data
#may be obtained by FutureBuf.read(), which is blocking.

#The gpu has two sequence buffers: A "small" buffer for MCMC gpu generation,
#and a "large buffer" for combined sequence sets.

#The GPU also has double buffers for Js and for Marginals ('front' and 'back').
#You can copy one buffer to the other with 'storebuf', and swap them with
#'swapbuf'.

#Note that in openCL implementations there is generally a limit on the
#number of queued items allowed in a context. If you reach the limit, all queues
#will block until a kernel finishes. So you must be careful that one GPU
#does not hog the queues.

#Note that on some systems there is a watchdog timer that kills any kernel
#that takes too long to finish. You will get a CL_OUT_OF_RESOURCES error
#if this happens, which occurs when the *following* kernel is run.

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
    def __init__(self, gpuinfo, (L, nB), nseq, wgsize, outdir, 
                 nhist, vsize, profile=False):
        self.L = L
        self.nB = nB
        self.nPairs = L*(L-1)/2
        self.events = []
        self.SWORDS = ((L-1)/4+1)     #num words needed to store a sequence
        self.SBYTES = (4*self.SWORDS) #num bytes needed to store a sequence
        self.nseq = {'main': nseq}

        gpu, gpunum, ctx, prg = gpuinfo
        self.gpunum = gpunum
        self.ctx = ctx
        self.prg = prg

        self.wgsize = wgsize
        self.nhist = nhist
        self.vsize = vsize

        self.logfn = os.path.join(outdir, 'gpu-{}.log'.format(gpunum))
        with open(self.logfn, "wt") as f:
            printDevice(f.write, gpu)

        self.mcmcprg = prg.metropolis

        #setup opencl for this device
        self.log("Getting CL Queue")
        self.profile = profile
        if profile:
            qprop = cl.command_queue_properties.PROFILING_ENABLE
            self.queue = cl.CommandQueue(ctx, device=gpu, properties=qprop)
        else:
            self.queue = cl.CommandQueue(ctx, device=gpu)
        self.log("\nOpenCL Device Compilation Log:")
        self.log(self.prg.get_build_info(gpu, cl.program_build_info.LOG))
        maxwgs = self.mcmcprg.get_work_group_info(
                 cl.kernel_work_group_info.WORK_GROUP_SIZE, gpu)
        self.log("Max MCMC WGSIZE: {}".format(maxwgs))
        
        self.initted = []

        self.bufs = {}
        self.buf_spec = {}
        self.Jbufs = {}
        self.bibufs = {}
        self.seqbufs = {}
        self.Ebufs = {}
        self.largebufs = []

        # setup generally needed buffers
        nPairs, SWORDS = self.nPairs, self.SWORDS
        self._setupBuffer( 'Jpacked', '<f4', (L*L, nB*nB))
        self._setupBuffer(  'J main', '<f4', (nPairs, nB*nB))
        self._setupBuffer( 'bi main', '<f4', (nPairs, nB*nB)),
        self._setupBuffer( 'bicount', '<u4', (nPairs, nB*nB)),
        self._setupBuffer('seq main', '<u4', (SWORDS, self.nseq['main'])),
        self._setupBuffer(  'E main', '<f4', (self.nseq['main'],)),
        self.packedJ = None #use to keep track of which Jbuf is packed

    def log(self, str):
        #logs are rare, so just open the file every time
        with open(self.logfn, "at") as f:
            print(time.clock(), str, file=f)

    def saveEvt(self, evt, name, nbytes=None):
        # don't save events if not profiling.
        # note that saved events use up memory - free it using logprofile
        if not self.profile:
            return
        if len(self.events)%1000 == 0 and self.events != []:
            log("Warning: Over {} profiling events are not flushed "
                "(using up memory)".format(len(self.events)))
        if nbytes:
            self.events.append((evt, name, nbytes))
        else:
            self.events.append((evt, name))

    def logProfile(self):
        #XXX don't call this if there are uncompleted events
        if not self.profile:
            return
        with open(self.logfn, "at") as f:
            while self.events != []:
                dat = self.events.pop(0)
                evt, name, size = dat[0],dat[1],(dat[2] if len(dat)==3 else '')
                print("EVT", name, evt.profile.start, evt.profile.end,
                      size, file=f)

    def _setupBuffer(self, bufname, buftype, bufshape, 
                          flags=cf.READ_WRITE | cf.ALLOC_HOST_PTR):
        size = dtype(buftype).itemsize*product(bufshape)
        buf = cl.Buffer(self.ctx, flags, size=size)

        self.bufs[bufname] = buf
        self.buf_spec[bufname] = (buftype, bufshape, flags)
        
        # add it to convenience dicts if applicable
        names = bufname.split()
        if len(names) > 1:
            {'J': self.Jbufs, 'bi': self.bibufs, 'seq': self.seqbufs, 
             'E': self.Ebufs}[names[0]][names[1]] = buf

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

        self.setBuf('Bs', ones(self.nseq['main'], dtype='<f4'))
        self._initMCMC_RNG(nMCMCcalls)
        self.nsteps = int(nsteps)

    def initLargeBufs(self, nseq_large):
        self._initcomponent('Large')

        self.nseq['large'] = nseq_large
        self._setupBuffer('seq large', '<u4', (self.SWORDS, self.nseq['large']))
        self._setupBuffer(  'E large', '<f4', (self.nseq['large'],))

        self.largebufs.extend(['seq large', 'E large'])

        # it is important to zero out the large seq buffer, because
        # if it is partially full we may need to compute energy
        # over the padding sequences at the end to get a full wg.
        buf = self.bufs['seq large']
        cl.enqueue_fill_buffer(self.queue, buf, uint32(0), 0, buf.size)

    def initSubseq(self):
        self.require('Large')
        self._initcomponent('Subseq')
        self._setupBuffer('markpos', '<u1',  (self.SBYTES,), cf.READ_ONLY)
        self.markPos(zeros(self.SBYTES, '<u1'))

    # we may want to select replicas at a particular temperature
    def initMarkSeq(self):
        self._initcomponent('Markseq')

        self._setupBuffer( 'markseqs', '<i4',  (self.nseq['main'],))
        self.setBuf('markseqs', arange(self.nseq['main'], dtype='<i4'))
        self.nmarks = self.nseq['main']

    def initJstep(self):
        self.require('Large')
        self._initcomponent('Jstep')

        nPairs, nB = sef.nPairs, self.nB
        self._setupBuffer('bi target', '<f4',  (nPairs, nB*nB))
        self._setupBuffer(     'Creg', '<f4',  (nPairs, nB*nB))
        self._setupBuffer(  'weights', '<f4',  (self.nseq['large'],))
        self._setupBuffer(     'neff', '<f4',  (1,))

        self.largebufs.append('weights')
        self.initBackBufs()

    def initBackBufs(self):
        self._initcomponent('Back')

        self._setupBuffer(  'J front', '<f4',  (nPairs, nB*nB))
        self._setupBuffer(   'J back', '<f4',  (nPairs, nB*nB))
        self._setupBuffer( 'bi front', '<f4',  (nPairs, nB*nB))
        self._setupBuffer(  'bi back', '<f4',  (nPairs, nB*nB))

    def packSeqs(self, seqs):
        "converts seqs to uchars, padded to 32bits, assume GPU is little endian"

        bseqs = zeros((seqs.shape[0], self.SBYTES), dtype='<u1', order='C')
        bseqs[:,:self.L] = seqs
        mem = zeros((self.SWORDS, seqs.shape[0]), dtype='<u4', order='C')
        for i in range(self.SWORDS):
            mem[i,:] = bseqs.view(uint32)[:,i]
        return mem

    def unpackSeqs(self, mem):
        bseqs = zeros((mem.shape[1], self.SBYTES), dtype='<u1', order='C')
        for i in range(self.SWORDS): #undo memory rearrangement
            bseqs.view(uint32)[:,i] = mem[i,:]
        return bseqs[:,:self.L]

    def packJ(self, Jbufname):
        """convert from format where every row is a unique ij pair (L choose 2
        rows) to format with every pair, all orders (L^2 rows)."""

        # quit if J already loaded/packed
        if self.packedJ == Jbufname:
            return  

        self.log("packJ " + Jbufname)

        nB, nPairs = self.nB, self.nPairs
        J_dev = self.Jbufs[Jbufname]
        evt = self.prg.packfV(self.queue, (nPairs*nB*nB,), (nB*nB,),
                        J_dev, self.bufs['Jpacked'])
        self.saveEvt(evt, 'packJ')
        self.packedJ = Jbufname

    def _initMCMC_RNG(self, nMCMCcalls):
        self.require('MCMC')
        self.log("initMCMC_RNG")

        rsize = 2

        nsamples = uint64(2**40) #upper bound for # of rngs generated
        nseq = self.nseq['main']
        offset = uint64(nsamples*nseq*self.gpunum*rsize)
        # each gpu uses perStreamOffset*get_global_id(0)*vectorSize samples
        #                    (nsamples *      nseq      * vecsize)

        # read mwc64 docs for description of nsamples.
        # Num rng samples should be chosen such that 2**64/(rsize*nsamples) is
        # greater than # walkers. nsamples should be > #MC steps performed
        # per walker (which is nsteps*nMCMCcalls)
        if not (self.nsteps*nMCMCcalls < nsamples< 2**64/(rsize*self.wgsize)):
            raise Exception("RNG problem. RNGs may not be independent.")
        #if this is a problem rethink the value 2**40 above, or consider using
        #an rng with a greater period, eg the "Warp" generator.

        wgsize = self.wgsize
        while wgsize > nseq:
            wgsize = wgsize/2
        evt = self.prg.initRNG2(self.queue, (nseq,), (wgsize,),
                         self.bufs['rngstates'], offset, nsamples)
        self.saveEvt(evt, 'initMCMC_RNG')

    def runMCMC(self):
        """Performs a single round of mcmc sampling (nsteps MC steps)"""
        self.require('MCMC')
        self.log("runMCMC")

        nseq = self.nseq['main']
        nsteps = self.nsteps
        self.packJ('main')
        self.setBuf('randpos', randint(0, self.L, size=nsteps).astype('u4'))
        evt = self.mcmcprg(self.queue, (nseq,), (self.wgsize,), 
                          self.bufs['Jpacked'], self.bufs['rngstates'], 
                          self.bufs['randpos'], uint32(nsteps), 
                          self.Ebufs['main'], self.bufs['Bs'], 
                          self.seqbufs['main'])
        self.saveEvt(evt, 'mcmc')

    def measureFPerror(self, log, nloops=3):
        log("Measuring FP Error")
        for n in range(nloops):
            self.runMCMC()
            e1 = self.getBuf('E main').read()
            self.calcEnergies('main', 'main')
            e2 = self.getBuf('E main').read()
            log("Run", n, "Error:", mean((e1-e2)**2))
            log('    Final E MC', printsome(e1), '...')
            log("    Final E rc", printsome(e2), '...')

            seqs = self.getBuf('seq main').read()
            J = self.getBuf('J main').read()
            e3 = getEnergies(seqs, J)
            log("    Exact E", e3[:5])
            log("    Error:", mean([float((a-b)**2) for a,b in zip(e1, e3)]))

    def calcBicounts(self, seqbufname):
        self.log("calcBicounts " + seqbufname)
        L, nB, nPairs, nhist = self.L, self.nB, self.nPairs, self.nhist

        if seqbufname == 'main':
            nseq = self.nseq[seqbufname]
            buflen = nseq
        else:
            nseq = self.nstoredseqs
            buflen = self.nseq[seqbufname]
        seq_dev = self.seqbufs[seqbufname]

        localhist = cl.LocalMemory(nhist*nB*nB*dtype(uint32).itemsize)
        evt = self.prg.countBivariate(self.queue, (nPairs*nhist,), (nhist,), 
                     self.bufs['bicount'], 
                     uint32(nseq), seq_dev, uint32(buflen), localhist)
        self.events.append((evt, 'calcBicounts'))

    def calcMarkedBicounts(self):
        self.require('Markseq')

        self.log("calcMarkedBicounts ")
        L, nB, nPairs, nhist = self.L, self.nB, self.nPairs, self.nhist
        nseq = self.nseq['main']
        seq_dev = self.seqbufs['main']

        localhist = cl.LocalMemory(nhist*nB*nB*dtype(uint32).itemsize)
        evt = self.prg.countMarkedBivariate(
                     self.queue, (nPairs*nhist,), (nhist,),
                     self.bufs['bicount'], uint32(nseq),
                     self.bufs['markseqs'], seq_dev, localhist)
        self.events.append((evt, 'calcMarkedBicounts'))

    def calcEnergies(self, seqbufname, Jbufname):
        self.log("calcEnergies " + seqbufname + " " + Jbufname)

        energies_dev = self.Ebufs[seqbufname]
        seq_dev = self.seqbufs[seqbufname]
        buflen = self.nseq[seqbufname]

        if seqbufname == 'main':
            nseq = self.nseq[seqbufname]
        else:
            nseq = self.nstoredseqs
            # pad to be a multiple of wgsize (uses dummy seqs at end)
            nseq = nseq + self.wgsize - (nseq % self.wgsize)

        self.packJ(Jbufname)

        evt = self.prg.getEnergies(self.queue, (nseq,), (self.wgsize,),
                             self.bufs['Jpacked'], seq_dev, uint32(buflen),
                             energies_dev)
        self.events.append((evt, 'getEnergies'))

    # update front bimarg buffer using back J buffer and large seq buffer
    def perturbMarg(self):
        self.log("perturbMarg")
        self.calcWeights()
        self.wait()
        self.weightedMarg()

    def calcWeights(self):
        self.require('Jstep')
        self.log("getWeights")

        #overwrites weights, neff
        #assumes seqmem_dev, energies_dev are filled in
        nseq = self.nstoredseqs
        self.packJ('back')
        buflen = self.nseq['large']

        # pad to be a multiple of wgsize (uses dummy seqs at end)
        ns_pad = nseq + self.wgsize - (nseq % self.wgsize)

        evt = self.prg.perturbedWeights(self.queue, (ns_pad,), (self.wgsize,),
                       self.bufs['Jpacked'],
                       self.seqbufs['large'], uint32(buflen),
                       self.bufs['weights'], self.Ebufs['large'])
        self.saveEvt(evt, 'perturbedWeights')
        localarr = cl.LocalMemory(self.vsize*dtype(float32).itemsize)
        evt = self.prg.sumFloats(self.queue, (self.vsize,), (self.vsize,), 
                            self.bufs['weights'], self.bufs['neff'], 
                            uint32(nseq), localarr)
        self.saveEvt(evt, 'sumFloats')
    
    def weightedMarg(self):
        self.require('Jstep')
        self.log("weightedMarg")
        nB, L, nPairs, nhist = self.nB, self.L, self.nPairs, self.nhist

        #like calcBimarg, but only works on large seq buf, and also calculate
        #neff. overwites front bimarg buf. Uses weights_dev,
        #neff. Usually not used by user, but is called from
        #perturbMarg
        localhist = cl.LocalMemory(nhist*nB*nB*dtype(float32).itemsize)
        evt = self.prg.weightedMarg(self.queue, (nPairs*nhist,), (nhist,),
                        self.bibufs['front'], self.bufs['weights'],
                        self.bufs['neff'], uint32(self.nstoredseqs),
                        self.seqbufs['large'], uint32(self.nseq['large']),
                        localhist)
        self.events.append((evt, 'weightedMarg'))

    # updates front J buffer using back J and bimarg buffers
    def updateJ(self, gamma, pc):
        self.require('Jstep')
        self.log("updateJPerturb")
        nB, nPairs = self.nB, self.nPairs
        #find next highest multiple of wgsize, for num work units
        nworkunits = self.wgsize*((nPairs*nB*nB-1)//self.wgsize+1)
        evt = self.prg.updatedJ(self.queue, (nworkunits,), (self.wgsize,),
                                self.bibufs['target'], self.bibufs['back'],
                                float32(gamma), float32(pc), self.Jbufs['main'],
                                self.Jbufs['back'], self.Jbufs['front'])
        self.saveEvt(evt, 'updateJPerturb')
        if self.packedJ == 'front':
            self.packedJ = None
    
    # XXX updateJ_reg
    def updateJ_weightfn(self, gamma, pc, fn_lmbda):
        self.require('Jstep')
        self.log("updateJPerturb_weightfn")
        nB, nPairs = self.nB, self.nPairs
        evt = self.prg.updatedJ_weightfn(self.queue, (nPairs*nB*nB,), (nB*nB,), 
                                self.bibufs['target'], self.bibufs['back'], 
                                self.bufs['Creg'],
                                float32(gamma), float32(pc), float32(fn_lmbda),
                                self.Jbufs['back'], self.Jbufs['front'])
        self.saveEvt(evt, 'updateJPerturb_weightfn')
        if self.packedJ == 'front':
            self.packedJ = None

    def getBuf(self, bufname, truncateLarge=True):
        """get buffer data. truncateLarge means only return the
        computed part of the large buffer (rest may be uninitialized)"""

        self.log("getBuf " + bufname)
        bufspec = self.buf_spec[bufname]
        buftype, bufshape = bufspec[0], bufspec[1]
        mem = zeros(bufshape, dtype=buftype)
        evt = cl.enqueue_copy(self.queue, mem, self.bufs[bufname],
                              is_blocking=False)
        self.saveEvt(evt, 'getBuf', mem.nbytes)
        if bufname.split()[0] == 'seq':
            if bufname in self.largebufs and truncateLarge:
                nret = self.nstoredseqs
                return FutureBuf(mem, evt,
                                 lambda b: self.unpackSeqs(b)[:nret,:])
            return FutureBuf(mem, evt, self.unpackSeqs)

        if bufname in self.largebufs and truncateLarge:
            nret = self.nstoredseqs
            return FutureBuf(mem, evt, lambda b: b[:nret])

        return FutureBuf(mem, evt)

    def setBuf(self, bufname, buf):
        self.log("setBuf " + bufname)

        if bufname.split()[0] == 'seq':
            buf = self.packSeqs(buf)

        bufspec = self.buf_spec[bufname]
        buftype, bufshape = bufspec[0], bufspec[1]
        if not isinstance(buf, ndarray):
            buf = array(buf, dtype=buftype)
        assert(dtype(buftype) == buf.dtype)
        assert(bufshape == buf.shape) or (bufshape == (1,) and buf.size == 1)

        evt = cl.enqueue_copy(self.queue, self.bufs[bufname], buf,
                              is_blocking=False)
        self.saveEvt(evt, 'setBuf', buf.nbytes)
        #unset packedJ flag if we modified that J buf
        if bufname.split()[0] == 'J':
            if bufname.split()[1] == self.packedJ:
                self.packedJ = None

    def swapBuf(self, buftype):
        self.require('backBufs')
        self.log("swapBuf " + buftype)

        #update convenience dicts
        bufs = {'J': self.Jbufs, 'bi': self.bibufs}[buftype]
        bufs['front'], bufs['back'] = bufs['back'], bufs['front']
        #update self.bufs
        bufs, t = self.bufs, buftype
        bufs[t+' front'], bufs[t+' back'] = bufs[t+' back'], bufs[t+' front']
        #update packedJ
        if buftype == 'J':
            self.packedJ = {'front':  'back',
                             'back': 'front'}.get(self.packedJ, self.packedJ)

    def storeBuf(self, buftype):
        self.require('backBufs')
        self.log("storeBuf " + buftype)
        self.copyBuf(buftype+' front', buftype+' back')

    def copyBuf(self, srcname, dstname):
        self.log("copyBuf " + srcname + " " + dstname)
        assert(srcname.split()[0] == dstname.split()[0])
        assert(self.buf_spec[srcname][1] == self.buf_spec[dstname][1])
        srcbuf = self.bufs[srcname]
        dstbuf = self.bufs[dstname]
        evt = cl.enqueue_copy(self.queue, dstbuf, srcbuf)
        self.saveEvt(evt, 'copyBuf')
        if dstname.split()[0] == 'J' and self.packedJ == dstname.split()[1]:
            self.packedJ = None

    def markPos(self, marks):
        self.require('Subseq')

        marks = marks.astype('<u1')
        if len(marks) == self.L:
            marks.resize(self.SBYTES)
        self.setBuf('markpos', marks)

    def fillSeqs(self, startseq, seqbufname='main'):
        #write a kernel function for this?
        self.log("fillSeqs " + seqbufname)
        nseq = self.nseq[seqbufname]
        self.setBuf('seq '+seqbufname, tile(startseq, (nseq,1)))

    def storeSeqs(self):
        self.require('Large')

        offset = self.nstoredseqs
        self.log("storeSeqs " + str(offset))

        nseq = self.nseq['main']
        if offset + nseq > self.nseq['large']:
            raise Exception("cannot store seqs past end of large buffer")
        evt = self.prg.storeSeqs(self.queue, (nseq,), (self.wgsize,),
                           self.seqbufs['main'], self.seqbufs['large'],
                           uint32(self.nseq['large']), uint32(offset))
        self.nstoredseqs += nseq
        self.saveEvt(evt, 'storeSeqs')

    def markSeqs(self, mask):
        self.require('Markseq')

        marks = -ones(mask.shape, dtype='i4')
        inds = argwhere(mask)
        marks[inds] = arange(len(inds), dtype='i4')
        self.setBuf('markseqs', marks)
        self.nmarks = sum(mask)
        self.log("marked {} seqs".format(self.nmarks))

    def storeMarkedSeqs(self):
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
        evt = self.prg.storeMarkedSeqs(self.queue, (nseq,), (self.wgsize,),
                           self.seqbufs['main'], self.seqbufs['large'],
                           uint32(self.nseq['large']), uint32(offset),
                           self.bufs['markseqs'])

        self.nstoredseqs += newseq
        self.events.append((evt, 'storeMarkedSeqs'))

    def clearLargeSeqs(self):
        self.require('Large')
        self.nstoredseqs = 0

    def restoreSeqs(self):
        self.require('Large')
        self.log("restoreSeqs " + str(offset))
        nseq = self.nseq['main']

        if offset + nseq > self.nseq['large']:
            raise Exception("cannot get seqs past end of large buffer")
        if self.nstoredseqs < nseq:
            raise Exception("not enough seqs stored in large buffer")

        evt = self.prg.restoreSeqs(self.queue, (nseq,), (self.wgsize,),
                           self.seqbufs['main'], self.seqbufs['large'],
                           uint32(self.nseq['large']), uint32(offset))
        self.saveEvt(evt, 'restoreSeqs')

    def copySubseq(self, seqind):
        self.require('Subseq')
        self.log("copySubseq " + str(seqind))
        nseq = self.nseq['large']
        if seqind >= self.nseq['main']:
            raise Exception("given index is past end of main seq buffer")
        evt = self.prg.copySubseq(self.queue, (nseq,), (self.wgsize,),
                           self.seqbufs['main'], self.seqbufs['large'],
                           uint32(self.nseq['main']), uint32(seqind),
                           self.bufs['markpos'])
        self.saveEvt(evt, 'copySubseq')

    def wait(self):
        self.log("wait")
        self.queue.finish()

################################################################################
# Set up enviroment and some helper functions

printsome = lambda a: " ".join(map(str,a.flatten()[:5]))

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

def packJ_CPU(self, couplings):
    """convert from format where every row is a unique ij pair (L choose 2
    rows) to format with every pair, all orders (L^2 rows). Note that the
    GPU kernel packfV does the same thing faster"""
    
    L, nB = seqsize_from_param_shape(couplings.shape)
    fullcouplings = zeros((L*L,nB*nB), dtype='<f4', order='C')
    pairs = [(i,j) for i in range(L-1) for j in range(i+1,L)]
    for n,(i,j) in enumerate(pairs):
        c = couplings[n,:]
        fullcouplings[L*i + j,:] = c
        fullcouplings[L*j + i,:] = c.reshape((nB,nB)).T.flatten()
    return fullcouplings

################################################################################

def setupGPUs(scriptpath, scriptfile, param, log):
    outdir = param.outdir
    L, nB = param.L, param.nB
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

        for d in dev:
            try:
                plat, devices = platforms[p]
                gpu = devices.popitem()[1]
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

    #compile CL program
    options = [('nB', nB), ('L', L)]
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
        log("PTX length: ", len(p))
        with open(os.path.join(outdir, 'ptx{}'.format(n)), 'wt') as f:
            f.write(p)

    return (cl_ctx, cl_prg), gpudevices

def divideWalkers(nwalkers, ngpus, wgsize, log):
    n_max = (nwalkers-1)/ngpus + 1
    nwalkers_gpu = [n_max]*(ngpus-1) + [nwalkers - (ngpus-1)*n_max]
    if nwalkers % (ngpus*wgsize) != 0:
        log("Warning: number of MCMC walkers is not a multiple of "
            "wgsize*ngpus, so there are idle work units.")
    return nwalkers_gpu

def initGPU(devnum, (cl_ctx, cl_prg), device, nwalkers, param, log):
    outdir = param.outdir
    L, nB = param.L, param.nB
    profile = param.profile
    wgsize = param.wgsize

    # wgsize = OpenCL work group size for MCMC kernel.
    # (also for other kernels, although would be nice to uncouple them)
    if wgsize not in [1<<n for n in range(32)]:
        raise Exception("wgsize must be a power of two")

    vsize = 1024 #power of 2. Work group size for 1d vector operations.

    #Number of histograms used in counting kernels (power of two),
    #which is the maximum parallelization of the counting step.
    #Each hist is nB*nB floats/uints, or 256 bytes for nB=8.
    #Here, chosen such that they use up to 16kb total.
    # For nB=21, get 8 hists. For nB=8, get 64.
    nhist = 4096//(nB*nB)
    if nhist == 0:
        raise Exception("alphabet size too large to make histogram on gpu")
    nhist = 2**int(log2(nhist)) #closest power of two

    log("Starting GPU {}".format(devnum))
    gpu = MCMCGPU((device, devnum, cl_ctx, cl_prg), (L, nB), 
                  nwalkers, wgsize, outdir, vsize, nhist, profile=profile)
    return gpu


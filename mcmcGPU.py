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
#will block until a kernel finishes. So all code must be careful that one GPU
#does not hog the queues.

#Note that on some systems there is a watchdog timer that kills any kernel 
#that takes too long to finish. You will get a CL_OUT_OF_RESOURCES error 
#if this happens, which occurs when the *following* kernel is run.

#Note that MCMC generation is split between nloop and nsteps.  Restarting the
#metropolis kernel as the effect of recalculating the current energy from
#scratch, which re-zeros any floating point error that may build up during one
#kernel run.

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
    def __init__(self, (gpu, gpunum, ctx, prg), (L, nB), outdir, nseq_small, 
                 nseq_large, wgsize, vsize, nhist, nMCMCcalls, nsteps=1, 
                 gibbs=False, profile=False):

        self.L = L
        self.nB = nB
        self.nPairs = L*(L-1)/2
        self.wgsize = wgsize
        self.nhist = nhist
        self.vsize = vsize
        self.events = []
        self.gpunum = gpunum

        self.logfn = os.path.join(outdir, 'gpu-{}.log'.format(gpunum))
        with open(self.logfn, "wt") as f:
            printDevice(f.write, gpu)

        self.gibbs = gibbs
        if gibbs:
            self.mcmcprg = prg.gibbs
            rngdtype = '<u8'
            self.log("Using Gibbs sampler")
        else:
            self.mcmcprg = prg.metropolis
            # rngstates should be size of mwc64xvec2_state_t
            rngdtype = '<2u8'
            self.log("Using Metropolis-Hastings sampler")

        #setup opencl for this device
        self.prg = prg
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


        #allocate device memory
        self.log("\nAllocating device buffers")

        self.SWORDS = ((L-1)/4+1)     #num words needed to store a sequence
        self.SBYTES = (4*self.SWORDS) #num bytes needed to store a sequence
        self.nseq = {'small': nseq_small,
                     'large': nseq_large}
        nPairs, SWORDS = self.nPairs, self.SWORDS
        self.log("\nBuffers: {} small seqs, {} large seqs".format(nseq_small, nseq_large))

        self.nsteps = int(nsteps)
        

        self.buf_spec = {   'Jpacked': ('<f4',  (L*L, nB*nB)),
                             'J main': ('<f4',  (nPairs, nB*nB)),
                            'J front': ('<f4',  (nPairs, nB*nB)),
                             'J back': ('<f4',  (nPairs, nB*nB)),
                            'bi main': ('<f4',  (nPairs, nB*nB)),
                           'bi front': ('<f4',  (nPairs, nB*nB)),
                            'bi back': ('<f4',  (nPairs, nB*nB)),
                          'bi target': ('<f4',  (nPairs, nB*nB)),
                            'bicount': ('<u4',  (nPairs, nB*nB)),
                          'seq small': ('<u4',  (SWORDS, self.nseq['small'])),
                          'seq large': ('<u4',  (SWORDS, self.nseq['large'])),
                          'rngstates': (rngdtype, (self.nseq['small'],)),
                            'E small': ('<f4',  (self.nseq['small'],)),
                            'E large': ('<f4',  (self.nseq['large'],)),
                               'fixJ': ('<u4',  (nPairs,)),
                            'weights': ('<f4',  (self.nseq['large'],)),
                               'neff': ('<f4',  (1,)),
                            'randpos': ('<u4',  (self.nsteps,))}

        self.bufs = {}
        flags = cl.mem_flags.READ_WRITE | cl.mem_flags.ALLOC_HOST_PTR
        for bname,(buftype,bufshape) in self.buf_spec.iteritems():
            size = dtype(buftype).itemsize*product(bufshape)
            self.bufs[bname] = cl.Buffer(ctx, flags, size=size)
        
        #convenience dicts:
        def getBufs(bufname):
            bnames = [(n.split(),b) for n,b in self.bufs.iteritems()]
            return dict((n[1], b) for n,b in bnames if n[0] == bufname)
        self.Jbufs = getBufs('J')
        self.bibufs = getBufs('bi')
        self.seqbufs = getBufs('seq')
        self.Ebufs = getBufs('E')

        self.bufs['fixpos'] = cl.Buffer(ctx, cl.mem_flags.READ_ONLY, size=L)
        self.buf_spec['fixpos'] = ('<u1', (L,))
        self.setBuf('fixpos', zeros(L, '<u1'))


        self.packedJ = None #use to keep track of which Jbuf is packed
        #(This class keeps track of Jpacked internally)
        
        self.initRNG(nMCMCcalls, gibbs, log)

        self.log("Initialization Finished\n")

    def log(self, str):
        #logs are rare, so just open the file every time
        with open(self.logfn, "at") as f:
            print(time.clock(), str, file=f)

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

    #converts seqs to uchars, padded to 32bits, assume GPU is little endian
    def packSeqs(self, seqs):
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

    #convert from format where every row is a unique ij pair (L choose 2 rows)
    #to format with every pair, all orders (L^2 rows)
    #Note that the GPU kernel packfV does the same thing faster
    def packJ_CPU(self, couplings):
        L,nB = self.L, self.nB
        fullcouplings = zeros((L*L,nB*nB), dtype='<f4', order='C')
        pairs = [(i,j) for i in range(L-1) for j in range(i+1,L)]
        for n,(i,j) in enumerate(pairs):
            c = couplings[n,:]
            fullcouplings[L*i + j,:] = c
            fullcouplings[L*j + i,:] = c.reshape((nB,nB)).T.flatten()
        return fullcouplings
    
    def packJ(self, Jbufname):
        if self.packedJ == Jbufname:
            return
        self.log("packJ " + Jbufname)

        nB, nPairs = self.nB, self.nPairs
        J_dev = self.Jbufs[Jbufname]
        evt = self.prg.packfV(self.queue, (nPairs*nB*nB,), (nB*nB,), 
                        J_dev, self.bufs['Jpacked'])
        self.events.append((evt, 'packJ'))
        self.packedJ = Jbufname

    def initRNG(self, nMCMCcalls, gibbs, log):
        self.log("initRNG")

        if gibbs:
            vsize = 1
            initkernel = self.prg.initRNG
        else:
            vsize = 2
            initkernel = self.prg.initRNG2

        nsamples = uint64(2**40) #upper bound for # of rngs generated
        nseq = self.nseq['small']
        offset = uint64(nsamples*nseq*self.gpunum*vsize) 
        # each gpu uses perStreamOffset*get_global_id(0)*vectorSize samples
        #               (nsamples *   nseq         *    vecsize)
        
        # read mwc64 docs for description of nsamples.
        # Num rng samples should be chosen such that 2**64/(vsize*nsamples) is
        # greater than # walkers. nsamples should be > #MC steps performed
        # per walker (which is nsteps*nMCMCcalls)
        if not (self.nsteps*nMCMCcalls < nsamples< 2**64/(vsize*self.wgsize)):
            log("Warning: RNG sampling problem. RNGs may not be independent.")
        #if this is a problem rethink the value 2**40 above, or consider using
        #an rng with a greater period, eg the "Warp" generator.

        wgsize = self.wgsize
        while wgsize > nseq:
            wgsize = wgsize/2
        evt = initkernel(self.queue, (nseq,), (wgsize,), 
                         self.bufs['rngstates'], offset, nsamples)
        self.events.append((evt, 'initRNG'))

    def runMCMC(self):
        self.log("runMCMC")
        nseq = self.nseq['small']
        nsteps = self.nsteps
        self.packJ('main')
        self.setBuf('randpos', randint(0, self.L, size=nsteps).astype('u4'))
        evt = self.mcmcprg(self.queue, (nseq,), (self.wgsize,), 
                          self.bufs['Jpacked'], self.bufs['rngstates'], 
                          self.bufs['randpos'], uint32(nsteps), 
                          self.Ebufs['small'], self.seqbufs['small'])
        self.events.append((evt, 'mcmc'))

    def measureFPerror(self, log, nloops=3):
        log("Measuring FP Error")
        for n in range(nloops):
            self.runMCMC()
            e1 = self.getBuf('E small').read()
            self.calcEnergies('small', 'main')
            e2 = self.getBuf('E small').read()
            log("Run", n, "Error:", mean((e1-e2)**2))
            log('    Final E MC', printsome(e1), '...')
            log("    Final E rc", printsome(e2), '...')

            seqs = self.getBuf('seq small').read()
            J = self.getBuf('J main').read()
            e3 = getEnergies(seqs, J)
            log("    Exact E", e3[:5])
            log("    Error:", mean([float((a-b)**2) for a,b in zip(e1, e3)]))

    def calcBimarg(self, seqbufname):
        self.log("calcBimarg " + seqbufname)
        L, nB, nPairs, nhist = self.L, self.nB, self.nPairs, self.nhist

        nseq = self.nseq[seqbufname]
        seq_dev = self.seqbufs[seqbufname]

        localhist = cl.LocalMemory(nhist*nB*nB*dtype(uint32).itemsize)
        evt = self.prg.countBimarg(self.queue, (nPairs*nhist,), (nhist,), 
                     self.bufs['bicount'], self.bibufs['main'], 
                     uint32(nseq), seq_dev, localhist)
        self.events.append((evt, 'calcBimarg'))

    def calcEnergies(self, seqbufname, Jbufname):
        self.log("calcEnergies " + seqbufname + " " + Jbufname)

        energies_dev = self.Ebufs[seqbufname]
        seq_dev = self.seqbufs[seqbufname]
        nseq = self.nseq[seqbufname]
        self.packJ(Jbufname)
        evt = self.prg.getEnergies(self.queue, (nseq,), (self.wgsize,), 
                             self.bufs['Jpacked'], seq_dev, energies_dev)
        self.events.append((evt, 'getEnergies'))

    # update front bimarg buffer using back J buffer and large seq buffer
    def perturbMarg(self): 
        self.log("perturbMarg")
        self.calcWeights()
        self.wait()
        self.weightedMarg()

    def calcWeights(self): 
        self.log("getWeights")

        #overwrites weights, neff
        #assumes seqmem_dev, energies_dev are filled in
        nseq = self.nseq['large']
        self.packJ('back')

        evt = self.prg.perturbedWeights(self.queue, (nseq,), (self.wgsize,), 
                       self.bufs['Jpacked'], self.seqbufs['large'],
                       self.bufs['weights'], self.Ebufs['large'])
        self.events.append((evt, 'perturbedWeights'))
        localarr = cl.LocalMemory(self.vsize*dtype(float32).itemsize)
        evt = self.prg.sumWeights(self.queue, (self.vsize,), (self.vsize,), 
                            self.bufs['weights'], self.bufs['neff'], 
                            uint32(nseq), localarr)
        self.events.append((evt, 'sumWeights'))
    
    def weightedMarg(self):
        self.log("weightedMarg")
        nB, L, nPairs, nhist = self.nB, self.L, self.nPairs, self.nhist

        #like calcBimarg, but only works on large seq buf, and also calculate
        #neff. overwites front bimarg buf. Uses weights_dev,
        #neff. Usually not used by user, but is called from
        #perturbMarg
        localhist = cl.LocalMemory(nhist*nB*nB*dtype(float32).itemsize)
        evt = self.prg.weightedMarg(self.queue, (nPairs*nhist,), (nhist,),
                        self.bibufs['front'], self.bufs['weights'], 
                        self.bufs['neff'], uint32(self.nseq['large']), 
                        self.seqbufs['large'], localhist)
        self.events.append((evt, 'weightedMarg'))

    # updates front J buffer using back J and bimarg buffers, possibly clamped
    # to orig coupling
    def updateJPerturb(self, gamma, pc, jclamp):
        self.log("updateJPerturb")
        nB, nPairs = self.nB, self.nPairs
        #find next highest multiple of wgsize, for num work units
        nworkunits = self.wgsize*((nPairs*nB*nB-1)//self.wgsize+1)
        evt = self.prg.updatedJ(self.queue, (nworkunits,), (self.wgsize,), 
                                self.bibufs['target'], self.bibufs['back'], 
                                float32(gamma), float32(pc), 
                                self.Jbufs['main'], float32(jclamp),
                                self.Jbufs['back'], self.Jbufs['front'])
        self.events.append((evt, 'updateJPerturb'))
        if self.packedJ == 'front':
            self.packedJ = None

    # updates front J buffer using back J and bimarg buffers, possibly clamped
    # to orig coupling
    def updateJPerturb_reg(self, gamma, pc, lmbda, sigma):
        self.log("updateJPerturb_reg")
        nB, nPairs = self.nB, self.nPairs
        evt = self.prg.updatedJ_reg(self.queue, (nPairs*nB*nB,), (nB*nB,), 
                                self.bibufs['target'], self.bibufs['back'], 
                                float32(gamma), float32(pc), float32(lmbda), float32(sigma),
                                self.Jbufs['back'], self.Jbufs['front'])
        self.events.append((evt, 'updateJPerturb_reg'))
        if self.packedJ == 'front':
            self.packedJ = None

    def updateJPerturb_fix(self, gamma, pc):
        self.log("updateJPerturb_fix")
        nB, nPairs = self.nB, self.nPairs
        evt = self.prg.updatedJ_fix(self.queue, (nPairs*nB*nB,), (nB*nB,), 
                                self.bibufs['target'], self.bibufs['back'], 
                                float32(gamma), float32(pc), self.bufs['fixJ'],
                                self.Jbufs['back'], self.Jbufs['front'])
        self.events.append((evt, 'updateJPerturb_fix'))
        if self.packedJ == 'front':
            self.packedJ = None

    def updateJ_weightfn(self, gamma, pc, fn_gamma, fn_s):
        self.log("updateJPerturb_weightfn")
        nB, nPairs = self.nB, self.nPairs
        evt = self.prg.updatedJ_weightfn(self.queue, (nPairs*nB*nB,), (nB*nB,), 
                                self.bibufs['target'], self.bibufs['back'], 
                                float32(gamma), float32(pc), 
                                float32(fn_gamma), float32(fn_s),
                                self.Jbufs['back'], self.Jbufs['front'])
        self.events.append((evt, 'updateJPerturb_weightfn'))
        if self.packedJ == 'front':
            self.packedJ = None

    def getBuf(self, bufname):
        self.log("getBuf " + bufname)
        buftype, bufshape = self.buf_spec[bufname]
        mem = zeros(bufshape, dtype=buftype)
        evt = cl.enqueue_copy(self.queue, mem, self.bufs[bufname], 
                              is_blocking=False)
        self.events.append((evt, 'getBuf', mem.nbytes))
        if bufname.split()[0] == 'seq':
            return FutureBuf(mem, evt, self.unpackSeqs)
        return FutureBuf(mem, evt)

    def setBuf(self, bufname, buf):
        self.log("setBuf " + bufname)

        if bufname.split()[0] == 'seq':
            buf = self.packSeqs(buf)

        buftype, bufshape = self.buf_spec[bufname]
        if not isinstance(buf, ndarray):
            buf = array(buf, dtype=buftype)
        assert(dtype(buftype) == buf.dtype)
        assert(bufshape == buf.shape) or (bufshape == (1,) and buf.size == 1)

        evt = cl.enqueue_copy(self.queue, self.bufs[bufname], buf, 
                              is_blocking=False)
        self.events.append((evt, 'setBuf', buf.nbytes))
        
        #unset packedJ flag if we modified that J buf
        if bufname.split()[0] == 'J':
            if bufname.split()[1] == self.packedJ:
                self.packedJ = None

    def swapBuf(self, buftype):
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
        self.log("storeBuf " + buftype)
        self.copyBuf(buftype+' front', buftype+' back')

    def copyBuf(self, srcname, dstname):
        self.log("copyBuf " + srcname + " " + dstname)
        assert(srcname.split()[0] == dstname.split()[0])
        assert(self.buf_spec[srcname][1] == self.buf_spec[dstname][1])
        srcbuf = self.bufs[srcname]
        dstbuf = self.bufs[dstname]
        evt = cl.enqueue_copy(self.queue, dstbuf, srcbuf)
        self.events.append((evt, 'copyBuf'))
        if dstname.split()[0] == 'J' and self.packedJ == dstname.split()[1]:
            self.packedJ = None

    def fillSeqs(self, startseq, seqbufname='small'):
        #write a kernel function for this?
        self.log("fillSeqs " + seqbufname)
        nseq = self.nseq[seqbufname]
        self.setBuf('seq '+seqbufname, tile(startseq, (nseq,1)))

    def storeSeqs(self, offset=0):
        self.log("storeSeqs " + str(offset))
        nseq = self.nseq['small']
        if offset + nseq > self.nseq['large']:
            raise Exception("cannot store seqs past end of large buffer")
        evt = self.prg.storeSeqs(self.queue, (nseq,), (self.wgsize,), 
                           self.seqbufs['small'], self.seqbufs['large'], 
                           uint32(self.nseq['large']), uint32(offset))
        self.events.append((evt, 'storeSeqs'))

    def restoreSeqs(self, offset=0):
        self.log("restoreSeqs " + str(offset))
        nseq = self.nseq['small']
        if offset + nseq > self.nseq['large']:
            raise Exception("cannot get seqs past end of large buffer")
        evt = self.prg.restoreSeqs(self.queue, (nseq,), (self.wgsize,), 
                           self.seqbufs['small'], self.seqbufs['large'], 
                           uint32(self.nseq['large']), uint32(offset))
        self.events.append((evt, 'restoreSeqs'))

    def copySubseq(self, seqind):
        self.log("copySubseq " + str(seqind))
        nseq = self.nseq['large']
        if seqind >= self.nseq['small']:
            raise Exception("given index is past end of small seq buffer")
        evt = self.prg.copySubseq(self.queue, (nseq,), (self.wgsize,), 
                           self.seqbufs['small'], self.seqbufs['large'], 
                           uint32(self.nseq['small']), uint32(seqind),
                           self.bufs['fixpos'])
        self.events.append((evt, 'copySubseq'))

    def wait(self):
        self.log("wait")
        self.queue.finish()

################################################################################
# Set up enviroment and some helper functions

printsome = lambda a: " ".join(map(str,a.flatten()[-5:]))

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

def initGPU(devnum, (cl_ctx, cl_prg), device, nwalkers, nlargebuf, param, log):
    outdir = param.outdir
    L, nB = param.L, param.nB
    nsteps  = param.nsteps
    profile = param.profile
    rngPeriod = param.rngPeriod
    wgsize = param.wgsize
    gibbs = param.gibbs

    # wgsize = OpenCL work group size for MCMC kernel. 
    # (also for other kernels, although would be nice to uncouple them)
    if wgsize not in [1<<n for n in range(32)]:
        raise Exception("wgsize must be a power of two")

    vsize = 256 #power of 2. Work group size for 1d vector operations.

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
    gpu = MCMCGPU((device, devnum, cl_ctx, cl_prg), (L, nB), outdir,
                  nwalkers, nlargebuf, wgsize, vsize, 
                  nhist, rngPeriod, nsteps, gibbs=gibbs, profile=profile)
    return gpu


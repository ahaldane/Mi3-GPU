#!/usr/bin/env python2
from scipy import *
import scipy
import numpy as np
from numpy.random import randint
import numpy.random
import pyopencl as cl
import pyopencl.array as cl_array
import sys, os, errno, glob, argparse, time
import seqload
from scipy.optimize import leastsq
from changeGauge import zeroGauge, zeroJGauge, fieldlessGauge

################################################################################
# Set up enviroment and some helper functions

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc: 
        if not (exc.errno == errno.EEXIST and os.path.isdir(path)):
            raise
scriptPath = os.path.dirname(os.path.realpath(__file__))
outdir = 'output'
printsome = lambda a: " ".join(map(str,a.flatten()[-5:]))

os.environ['PYOPENCL_COMPILER_OUTPUT'] = '0'
os.environ['PYOPENCL_NO_CACHE'] = '1'
os.environ["CUDA_CACHE_DISABLE"] = '1'

#identical calculation as CL kernel, but with high precision (to check fp error)
def getEnergies(s, couplings): 
    from mpmath import mpf, mp
    mp.dps = 32
    couplings = [[mpf(float(x)) for x in r] for r in couplings]
    pairenergy = [mpf(0) for n in range(s.shape[0])]
    for n,(i,j) in enumerate([(i,j) for i in range(L-1) for j in range(i+1,L)]):
        r = couplings[n]
        cpl = (r[b] for b in (nB*s[:,i] + s[:,j]))
        pairenergy = [x+n for x,n in zip(pairenergy, cpl)]
    return pairenergy

def printPlatform(p,n=0,f=sys.stdout):
    print >>f, "Platform {} '{}':".format(n, p.name)
    print >>f, "    Vendor: {}".format(p.vendor)
    print >>f, "    Version: {}".format(p.version)
    print >>f, "    Extensions: {}".format(p.extensions)

def printDevice(d, f=sys.stdout):
    print >>f, "  Device '{}':".format(d.name)
    print >>f, "    Vendor: {}".format(d.vendor)
    print >>f, "    Version: {}".format(d.version)
    print >>f, "    Driver Version: {}".format(d.driver_version)
    print >>f, "    Max Clock Frequency: {}".format(d.max_clock_frequency)
    print >>f, "    Max Compute Units: {}".format(d.max_compute_units)
    print >>f, "    Max Work Group Size: {}".format(d.max_work_group_size)
    print >>f, "    Global Mem Size: {}".format(d.global_mem_size)
    print >>f, "    Global Mem Cache Size: {}".format(d.global_mem_cache_size)
    print >>f, "    Local Mem Size: {}".format(d.local_mem_size)
    print >>f, "    Max Constant Buffer Size: {}".format(
                                                     d.max_constant_buffer_size)

def printGPUs():
    for n,p in enumerate(cl.get_platforms()):
        printPlatform(p,n)
        for d in p.get_devices():
            printDevice(d)
        print ""

################################################################################

#The GPU performs two types of computation: MCMC runs, and perturbed
#coupling updates.  All GPU methods are asynchronous. Functions that return
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

class MCMCGPU:
    def __init__(self, (gpu, gpunum, ctx, prg), bimarg_target, nseq_small,
                 nseq_large, wgsize, vsize, nhist, nsteps=1, profile=False):

        self.L = int(((1+sqrt(1+8*bimarg_target.shape[0]))/2) + 0.5) 
        self.nB = int(sqrt(bimarg_target.shape[1]) + 0.5) #+0.5 for rounding 
        self.nPairs = L*(L-1)/2
        self.wgsize = wgsize
        self.nhist = nhist
        self.vsize = vsize
        self.events = []
        self.gpunum = gpunum
        
        self.logfn = os.path.join(outdir, 'gpu-{}.log'.format(gpunum))
        with open(self.logfn, "wt") as f:
            printDevice(gpu, f)

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
        maxwgs = self.prg.metropolis.get_work_group_info(
                 cl.kernel_work_group_info.WORK_GROUP_SIZE, gpu)
        self.log("Max MCMC WGSIZE: {}".format(maxwgs))


        #allocate device memory
        self.log("\nAllocating device buffers")

        self.SWORDS = ((L-1)/4+1)     #num words needed to store a sequence
        self.SBYTES = (4*self.SWORDS) #num bytes needed to store a sequence
        self.nseq = {'small': nseq_small,
                      'large': nseq_large}
        nPairs, SWORDS = self.nPairs, self.SWORDS

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
                         # rngstates should be size of mwc64xvec2_state_t
                          'rngstates': ('<2u8', self.nseq['small']),
                            'E small': ('<f4',  self.nseq['small']),
                            'E large': ('<f4',  self.nseq['large']),
                            'weights': ('<f4',  self.nseq['large']),
                               'neff': ('<f4',  1) }

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

        self.packedJ = None #use to keep track of which Jbuf is packed
        #(This class keeps track of Jpacked internally)
        
        self.nsteps = nsteps*L #increase if L very small
        self.setBuf('bi target', bimarg_target)

        self.initRNG()

        self.log("Initialization Finished\n")

    def log(self, str):
        #logs are rare, so just open the file every time
        with open(self.logfn, "at") as f:
            print >>f, time.clock(), str

    def logProfile(self):
        #XXX don't call this if there are uncompleted events
        if not self.profile:
            return
        with open(self.logfn, "at") as f:
            while self.events != []:
                dat = self.events.pop(0)
                evt, name, size = dat[0],dat[1],(dat[2] if len(dat)==3 else '')
                print >>f, "EVT", name, evt.profile.start, evt.profile.end, size

    #converts seqs to uchars, padded to 32bits, assume GPU is little endian
    def packSeqs(self, seqs):
        bseqs = zeros((seqs.shape[0], self.SBYTES), dtype='<u1', order='C')
        bseqs[:,:L] = seqs  
        mem = zeros((self.SWORDS, seqs.shape[0]), dtype='<u4', order='C')
        for i in range(self.SWORDS):
            mem[i,:] = bseqs.view(uint32)[:,i]
        return mem

    def unpackSeqs(self, mem):
        bseqs = zeros((mem.shape[1], self.SBYTES), dtype='<u1', order='C')
        for i in range(self.SWORDS): #undo memory rearrangement
            bseqs.view(uint32)[:,i] = mem[i,:] 
        return bseqs[:,:L]

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

    def initRNG(self):
        self.log("initRNG")

        nsamples = uint64(2**40) #upper bound for # of uint2 rngs generated
        nseq = self.nseq['small']
        offset = uint64(nsamples*nseq*self.gpunum*2) 
        offset = uint64(nsamples*nseq*self.gpunum) #XXX temp 
        # each gpu uses perStreamOffset*get_global_id(0)*vectorSize samples
        #               (nsamples      *   nseq         *    2)
        
        # read mwc64 docs for description of nsamples.
        # Num samples should be chosen such that 2**64/(2*nsamples) is greater
        # than # walkers. nsamples should be > #MC steps performed per walker
        # (which is nsteps*L*nloop*gdsteps)
        if not (self.nsteps*nloop*gdsteps < nsamples < 2**64/(2*nwalkers)):
            print "Warning: RNG sampling problem. RNGs may not be independent."
        #if this is a problem rethink the value 2**40 above, or consider using
        #an rng with a greater period, eg the "Warp" generator.

        evt = self.prg.initRNG(self.queue, (nseq,), (self.wgsize,), 
                               self.bufs['rngstates'], offset, nsamples)
        self.events.append((evt, 'initRNG'))

    def runMCMC(self):
        self.log("runMCMC")
        nseq = self.nseq['small']
        self.packJ('main')
        evt = self.prg.metropolis(self.queue, (nseq,), (self.wgsize,), 
                            self.bufs['Jpacked'], self.bufs['rngstates'], 
                            uint32(self.nsteps), self.Ebufs['small'], 
                            self.seqbufs['small'])
        self.events.append((evt, 'metropolis'))

    def measureFPerror(self, nloops=3):
        print "Measuring FP Error"
        for n in range(nloops):
            self.runMCMC()
            e1 = self.getBuf('E small').read()
            self.calcEnergies('small', 'main')
            e2 = self.getBuf('E small').read()
            print "Run", n, "Error:", mean((e1-e2)**2)
            print '    Final E MC', printsome(e1), '...'
            print "    Final E rc", printsome(e2), '...'

            seqs = self.getBuf('seq small').read()
            J = self.getBuf('J main').read()
            e3 = getEnergies(seqs, J)
            print "    Exact E", e3[:5]
            print "    Error:", mean([float((a-b)**2) for a,b in zip(e1, e3)])

    def calcBimarg(self, seqbufname):
        self.log("calcBimarg " + seqbufname)
        L, nPairs, nhist = self.L, self.nPairs, self.nhist

        nseq = self.nseq[seqbufname]
        seq_dev = self.seqbufs[seqbufname]

        evt = self.prg.countBimarg(self.queue, (nPairs*nhist,), (nhist,), 
                     self.bufs['bicount'], self.bibufs['main'], 
                     uint32(nseq), seq_dev)
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
        evt = self.prg.sumWeights(self.queue, (self.vsize,), (self.vsize,), 
                            self.bufs['weights'], self.bufs['neff'], 
                            uint32(nseq))
        self.events.append((evt, 'sumWeights'))
    
    def weightedMarg(self):
        self.log("weightedMarg")
        nB, L, nPairs, nhist = self.nB, self.L, self.nPairs, self.nhist

        #like calcBimarg, but only works on large seq buf, and also calculate
        #neff. overwites front bimarg buf. Uses weights_dev,
        #neff. Usually not used by user, but is called from
        #perturbMarg
        evt = self.prg.weightedMarg(self.queue, (nPairs*nhist,), (nhist,),
                        self.bibufs['front'], self.bufs['weights'], 
                        self.bufs['neff'], uint32(self.nseq['large']), 
                        self.seqbufs['large'])
        self.events.append((evt, 'weightedMarg'))

    # updates front J buffer using back J and bimarg buffers, possibly clamped
    # to orig coupling
    def updateJPerturb(self, gamma):
        self.log("updateJPerturb")
        nB, nPairs = self.nB, self.nPairs
        #find next highest multiple of wgsize, for num work units
        nworkunits = self.wgsize*((nPairs*nB*nB-1)//self.wgsize+1)
        evt = self.prg.updatedJ(self.queue, (nworkunits,), (self.wgsize,), 
                          self.bibufs['target'], self.bibufs['back'], 
                          self.Jbufs['main'], float32(gamma), 
                          self.Jbufs['back'], self.Jbufs['front'])
        self.events.append((evt, 'updateJPerturb'))
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
        assert(buftype == buf.dtype.str)
        assert((bufshape == buf.shape) or (bufshape == 1 and buf.size == 1))

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

    def resetSeqs(self, startseq, seqbufname='small'):
        #write a kernel function for this?
        self.log("resetSeqs " + seqbufname)
        nseq = self.nseq[seqbufname]
        self.setBuf('seq '+seqbufname, tile(startseq, (nseq,1)))

    def storeSeqs(self, offset=0):
        self.log("storeSeqs " + str(offset))
        nseq = self.nseq['small']
        evt = self.prg.storeSeqs(self.queue, (nseq,), (self.wgsize,), 
                           self.seqbufs['small'], self.seqbufs['large'], 
                           uint32(offset*nseq))
        self.events.append((evt, 'storeSeqs'))

    def wait(self):
        self.log("wait")
        self.queue.finish()

################################################################################
# Read in args and Data:

#parse arguments
class CLInfoAction(argparse.Action):
    def __init__(self, option_strings, dest=argparse.SUPPRESS, 
                 default=argparse.SUPPRESS, help=None):
        super(CLInfoAction, self).__init__(option_strings=option_strings,
            dest=dest, default=default, nargs=0, help=help)
    def __call__(self, parser, namespace, values, option_string=None):
        printGPUs()
        parser.exit()

parser = argparse.ArgumentParser(description='GD-MCMC')
parser.add_argument('--clinfo', action=CLInfoAction)
parser.add_argument('bimarg')
parser.add_argument('gamma', type=float32)
parser.add_argument('gdsteps', type=uint32)
parser.add_argument('nwalkers', type=uint32)
parser.add_argument('nloop', type=uint32)
parser.add_argument('nsampleloops', type=uint32)
parser.add_argument('nsamples', type=uint32)
parser.add_argument('alpha', help="Alphabet, a sequence of letters")
parser.add_argument('-nsteps', type=uint32, default=1,
                    help="number of MC steps per loop, in multiples of L")
parser.add_argument('-wgsize', type=int, default=256)
parser.add_argument('-outdir', default='output')
parser.add_argument('-pc', default=0)
parser.add_argument('-pcdamping', default=0.001)
parser.add_argument('-Jcutoff', default=None)
parser.add_argument('-couplings', default='none', 
                    help="One of 'zero', 'logscore', or a filename")
parser.add_argument('-startseq', help="Either a sequence, or 'rand'") 
parser.add_argument('-start', default='none', 
                    help="A directory name")
parser.add_argument('-prestart', default='none', 
                    help="One of 'zero', 'logscore', or a directory name")
parser.add_argument('-trackequil', type=uint32, default=0,
                    help='during equilibration, save bimarg every N loops')
parser.add_argument('-gpus')
parser.add_argument('-benchmark', action='store_true')
parser.add_argument('-profile', action='store_true')
parser.add_argument('-measureFPerror', action='store_true')
parser.add_argument('-perturbSteps', default='128')
parser.add_argument('-regularizationScale', default=0.5)

args = parser.parse_args(sys.argv[1:])

print "Initialization\n==============="
print ""
print "Parameter Setup"
print "---------------"

outdir = args.outdir
mkdir_p(outdir)

print "Reading bimarg from file {}".format(args.bimarg)
bimarg_target = scipy.load(args.bimarg)
if bimarg_target.dtype != dtype('<f4'):
    raise Exception("Bimarg in wrong format")
    #could easily convert, but this helps warn that something may be wrong
if any(~( (bimarg_target.flatten() >= 0) & (bimarg_target.flatten() <= 1))):
    raise Exception("Bimarg must be nonzero and 0 < f < 1")
pc = float(args.pc)
if pc != 0:
    print "Adding pseudocount of {} to marginals".format(pc)
    bimarg_target = bimarg_target + pc
    bimarg_target = bimarg_target/sum(bimarg_target, axis=1)[:,newaxis]

L = int(((1+sqrt(1+8*bimarg_target.shape[0]))/2) + 0.5) 
nB = int(sqrt(bimarg_target.shape[1]) + 0.5) #+0.5 for rounding any fp error
nPairs = L*(L-1)/2;
n_couplings = nPairs*nB*nB
print "nBases {}  seqLen {}".format(nB, L)
gdsteps = args.gdsteps  # 10
print "Running {} GD steps".format(gdsteps)
print ""

                        # for example:
gamma0 = args.gamma     # 0.001
nloop = args.nloop      # 100
nsteps = args.nsteps    # 1 #increase this if L small. = num of MCMC iterations
nwalkers = args.nwalkers    
nsampleloops = args.nsampleloops    
nsamples = args.nsamples  

if nsamples == 0:
    raise Exception("nsamples must be at least 1")

print ("Running {} MC walkers for {} loops then sampling every {} loops to "
       "get {} samples ({} total seqs) with {} MC steps per loop (Each walker equilibrated a total of {} MC steps).").format(
           nwalkers, nloop, nsampleloops, nsamples, nsamples*nwalkers, 
           nsteps*L, nsteps*L*nloop),
if args.trackequil != 0:
    if nloop%args.trackequil != 0:
        raise Exception("Error: trackequil must be a divisor of nloop")
    print "Tracking equilibration every {} loops.".format(args.trackequil)
else:
    print ""
print ""


Jcutoff = args.Jcutoff
pcDamping = args.pcdamping
pSteps = [int(x) for x in args.perturbSteps.split(':')]
perturbSteps = (pSteps[0], pSteps[1] if len(pSteps) > 1 else 1)
regularizationScale = float(args.regularizationScale)


cutoffstr = 'dJ cutoff {}'.format(Jcutoff) if Jcutoff != None else 'no dJ cutoff'
print ("Updating J locally with gamma = {}, {}, and pc-damping {}. "
       "Running {} update steps.").format(gamma0, cutoffstr, pcDamping, 
                                          perturbSteps[0]*perturbSteps[1]),
       
if regularizationScale != 0:
    if perturbSteps[1] == 1:
        sstr = "once"
    else:
        sstr = "every {} J update steps".format(perturbSteps[0])
    print "Regularizing {} with a scale of {}".format(sstr, regularizationScale)
elif perturbSteps[1] != 1:
    print ("Warning: regularization scale is 0, yet regularization is "
           "requested every {} update steps").format(perturbSteps[1])
else:
    print ""
print ""

bicount = empty((nPairs, nB*nB), dtype='<u4')
alpha = args.alpha
if len(alpha) != nB:
    print "Expected alphabet size {}, got {}".format(nB, len(alpha))
    exit()

if sum([v == 'none' for v in [args.start, args.prestart]]) > 1:
    raise Exception("Can only specify one of: prestart start")
startinfo = args.prestart if args.prestart != 'none' else args.start

if args.couplings != 'none':
    couplinginfo = args.couplings
elif args.prestart in ['zero', 'logscore']:
    couplinginfo = args.prestart
elif args.start != 'none' or args.prestart != 'none':
    couplinginfo = os.path.join(startinfo, 'J.npy')
else:
    raise Exception("Must supply one of: start, prestart, couplings")

if couplinginfo == 'zero':
    print "Setting Initial couplings to 0"
    couplings = zeros((nPairs, nB*nB), dtype='<f4')
elif couplinginfo == 'logscore':
    print "Setting Initial couplings to Independent Log Scores"
    ff = bimarg_target.reshape((nPairs,nB,nB))
    marg = array([sum(ff[0],axis=1)] + [sum(ff[n],axis=0) for n in range(L-1)])
    marg = marg/(sum(marg,axis=1)[:,newaxis]) # correct any fp errors
    h = -log(marg)
    h = h - mean(h, axis=1)[:,newaxis]
    couplings = fieldlessGauge(h, zeros((nPairs,nB*nB),dtype='<f4'))[1]
else:
    print "Reading initial couplings from file {}".format(couplinginfo)
    couplings = scipy.load(couplinginfo)
    if couplings.dtype != dtype('<f4'):
        raise Exception("Couplings in wrong format")
#switch to 'even' fieldless gauge for nicer output
h0, J0 = zeroGauge(zeros((L,nB)), couplings)
couplings = fieldlessGauge(h0, J0)[1]
save(os.path.join(outdir, 'startJ'), couplings)

if args.startseq:
    if args.startseq == 'rand':
        startseq = randint(0, nB, size=L).astype('<u1')
    else:
        startseq = array([alpha.index(c) for c in args.startseq], dtype='<u1')
elif args.start != 'none' or args.prestart != 'none':
    if args.start != 'none' or (args.prestart not in ['logscore', 'zero']):
        fn = os.path.join(startinfo, 'startseq')
        print "Reading startseq from file {}".format(fn)
        with open(fn) as f:
            startseq = f.readline().strip()
            startseq = array([alpha.index(c) for c in startseq], dtype='<u1')
    else:
        print "Start seq taken as first generated sequence during pre-opt"
        startseq = None
else:
    raise Exception("Must supply one of: start, prestart, startseq")
sstype = 'random' if args.startseq == 'rand' else 'provided'

print ""
print "Target Marginals: " + printsome(bimarg_target) + "..."
print "Initial Couplings: " + printsome(couplings) + "..."
if startseq is not None:
    print "Start seq ({}): {}".format(sstype, "".join(alpha[x] 
                                                      for x in startseq))
else:
    print "Start seq: To be generated during pre-optimization"

     

################################################################################
#setup gpus

print ""
print "GPU setup"
print "---------"

with open(os.path.join(scriptPath, "metropolis.cl")) as f:
    src = f.read()

#figure out which gpus to use
gpudevices = []
platforms = [(p, p.get_devices()) for p in cl.get_platforms()]
if args.gpus:
    gpustr = args.gpus
    #parse user arg
    try:
        dev = [tuple(int(x) for x in a.split(':')) for a in gpustr.split(',')]
    except:
        raise Exception("Error: GPU specification must be comma separated "
                        " list of platforms, eg '0,0'")

    for p in dev:
        try:
            plat, devices = platforms[p]
            gpu = devices.popitem()[1]
        except:
            raise Exception("No GPU with specification {}".format(p))
        print "Using GPU {} on platform {} ({})".format(gpu.name, plat.name, p)
        gpudevices.append(gpu)
else:
    #use gpus in first platform
    plat = platforms[0]
    for gpu in plat[1]:
        print "Using GPU {} on platform {} (0)".format(gpu.name, plat[0].name)
        gpudevices.append(gpu)

if len(gpudevices) == 0:
    raise Exception("Error: No GPUs found")

#set up OpenCL. Assumes all gpus are identical
print "Getting CL Context"
cl_ctx = cl.Context(gpudevices)

#divide up seqs to gpus
wgsize = args.wgsize #OpenCL work group size for MCMC kernel. 
if wgsize not in [1<<n for n in range(32)]:
    raise Exception("wgsize must be a power of two")
if nwalkers % (len(gpudevices)*wgsize) != 0:
    raise Exception("nwalkers must be mutliple of wgsize*ngpus")
nwalkers_gpu = nwalkers/len(gpudevices)

vsize = 256 #power of 2. Work group size for 1d vector operations.
nhist = 64 #power of two. Number of histograms used in counting 
           #kernels (each hist is nB*nB floats/uints). Should really be
           # made a function of nB, eg 4096/(nB*nB) for occupancy ~ 3

#compile CL program
options = [('WGSIZE', wgsize), ('NSEQS', nwalkers_gpu), ('NSAMPLES', nsamples),
           ('VSIZE', vsize), ('NHIST', nhist), ('nB', nB), ('L', L), 
           ('PC', pcDamping)]
if args.measureFPerror:
    options.append(('MEASURE_FP_ERROR', 1))
if Jcutoff:
    options.append(('JCUTOFF', Jcutoff))
optstr = " ".join(["-D {}={}".format(opt,val) for opt,val in options]) 
print "Compilation Options: ", optstr
extraopt = " -cl-nv-verbose -Werror -I {}".format(scriptPath)
print "Compiling CL..."
cl_prg = cl.Program(cl_ctx, src).build(optstr + extraopt) 
#dump compiled program
ptx = cl_prg.get_info(cl.program_info.BINARIES)
for n,p in enumerate(ptx):
    #useful to see if compilation changed
    print "PTX length: ", len(p)
    with open(os.path.join(outdir, 'ptx{}'.format(n)), 'wt') as f:
        f.write(p)

gpus = []
profile = args.profile
print "Initializing Devices..."
for devnum, device in enumerate(gpudevices):
    gpu = MCMCGPU((device, devnum, cl_ctx, cl_prg), bimarg_target,
                  nwalkers_gpu, nsamples*nwalkers_gpu, wgsize, vsize, nhist, 
                  nsteps, profile=profile)
    gpus.append(gpu)

################################################################################
#Helper funcs

def writeStatus(name, rmsd, ssr, wdf, bicount, bimarg_model, couplings, 
                seqs, startseq, energies):

    #print some details 
    disp = ["Start Seq: " + "".join([alpha[c] for c in startseq]),
            "RMSD: {: 9.7f}  SSR: {: 9.5f}  wDf: {: 9.5f}".format(rmsd,ssr,wdf),
            "Bicounts: " + printsome(bicount) + '...',
            "Marginals: " + printsome(bimarg_model) + '...',
            "Couplings: " + printsome(couplings) + "...",
            "Energies: Lowest =  {}, Mean = {}".format(min(energies), 
                                                       mean(energies))]
    dispstr = "\n".join(disp)
    with open(os.path.join(outdir, name, 'info.txt'), 'wt') as f:
        print >>f, dispstr

    #save current state to file
    savetxt(os.path.join(outdir, name, 'bicounts'), bicount, fmt='%d')
    save(os.path.join(outdir, name, 'bimarg'), bimarg_model)
    save(os.path.join(outdir, name, 'energies'), energies)
    for n,seqbuf in enumerate(seqs):
        seqload.writeSeqs(os.path.join(outdir, name, 'seqs-{}'.format(n)), 
                          seqbuf, alpha)

    print dispstr

def sumarr(arrlist):
    #low memory usage (rather than sum(arrlist, axis=0))
    tot = arrlist[0].copy()
    for a in arrlist[1:]:
        np.add(tot, a, tot)
    return tot

def meanarr(arrlist):
    return sumarr(arrlist)/len(arrlist)

def readGPUbufs(bufnames, gpus):
    futures = [[gpu.getBuf(bn) for gpu in gpus] for bn in bufnames]
    return [[buf.read() for buf in gpuf] for gpuf in futures]

################################################################################
#Functions which perform the computation

def doFit(startseq, couplings, gpus):
    for i in range(gdsteps):
        runname = 'run_{}'.format(i)
        startseq, couplings = singleStep(runname, couplings, startseq, gpus)

def singleStep(runName, couplings, startseq, gpus):
    print ""
    print "Gradient Descent step {}".format(runName)

    mkdir_p(os.path.join(outdir, runName))
    save(os.path.join(outdir, runName, 'J'), couplings)
    with open(os.path.join(outdir, runName, 'startseq'), 'wt') as f:
        f.write("".join(alpha[c] for c in startseq))

    #get ready for MCMC
    for gpu in gpus:
        gpu.resetSeqs(startseq)
        gpu.setBuf('J main', couplings)
    
    #equilibration MCMC
    if args.trackequil == 0:
        #keep nloop iterator on outside to avoid filling queue with only 1 gpu
        for i in range(nloop):
            for gpu in gpus:
                gpu.runMCMC()
    else:
        #note: sync necessary with trackequil (may slightly affect performance)
        mkdir_p(os.path.join(outdir, runName, 'equilibration'))
        #mkdir_p(os.path.join(outdir, runName, 'eqseqs'))
        #mkdir_p(os.path.join(outdir, runName, 'eqen'))
        for j in range(nloop/args.trackequil):
            for i in range(args.trackequil):
                for gpu in gpus:
                    gpu.runMCMC()
            for gpu in gpus:
                gpu.calcBimarg('small')
                #gpu.calcEnergies('small', 'main')
            #res = readGPUbufs(['bi main', 'seq small', 'E small'], gpus)
            #res = readGPUbufs(['bi main', 'seq small'], gpus)
            #bimarg_model = meanarr(res[0])
            #pseq = res[1]
            #for n,seqbuf in enumerate(pseq):
            #    #save(os.path.join(outdir, runName, 'eqen', 'en-{}-{}'.format(n,j)), res[2][n])
            #    seqload.writeSeqs(os.path.join(outdir, runName, 'eqseqs', 'seqs-{}-{}'.format(n, j)), seqbuf, alpha)
            bimarg_model = meanarr(readGPUbufs(['bi main'], gpus)[0])
            save(os.path.join(outdir, runName, 
                 'equilibration', 'bimarg_{}'.format(j)), bimarg_model)

    #post-equilibration samples
    for gpu in gpus:
        gpu.storeSeqs(offset=0) #save seqs from smallbuf to largebuf
    for j in range(1,nsamples):
        for i in range(nsampleloops):
            for gpu in gpus:
                gpu.runMCMC()
        for gpu in gpus:
            gpu.storeSeqs(offset=j)
    
    #process results
    for gpu in gpus:
        gpu.calcBimarg('large')
        gpu.calcEnergies('large', 'main')
    res = readGPUbufs(['bi main', 'bicount', 'E large', 'seq large'], gpus)
    bimarg_model, bicount = meanarr(res[0]), sumarr(res[1])
    sampledenergies, sampledseqs = concatenate(res[2]), res[3]

    #get summary statistics and output them
    rmsd = sqrt(mean((bimarg_target - bimarg_model)**2))
    ssr = sum((bimarg_target - bimarg_model)**2)
    wdf = sum(bimarg_target*abs(bimarg_target - bimarg_model))
    writeStatus(runName, rmsd, ssr, wdf, bicount, bimarg_model, 
                couplings, sampledseqs, startseq, sampledenergies)
    
    #compute new J using local optimization
    couplings, bimarg_p = localDescent(perturbSteps, gamma0, gpus)
    save(os.path.join(outdir, runName, 'predictedBimarg'), bimarg_p)

    #choose seed sequence for next round
    rseq_ind = numpy.random.randint(0, len(sampledenergies))
    nseq = gpus[0].nseq['large'] #assumes all gpus the same
    rseq = sampledseqs[rseq_ind/nseq][rseq_ind%nseq]

    return rseq, couplings

################################################################################
#local optimization related code

def Jbias(J, scale=0.5):
    h0, J0 = zeroJGauge(zeros((L,nB)), J)
    #J0 = J0*(1-exp(-abs(J0)/(scale*std(J0.flatten()))))
    fb = sqrt(sum(J0**2, axis=1))
    J0 = J0*((1-exp(-fb/(scale*mean(fb))))[:,newaxis])
    return fieldlessGauge(h0, J0)[1]

def localStep(n, gamma, lastssr, gpus):
    #calculate perturbed marginals
    for gpu in gpus:
        #note: updateJPerturb should give same result on all GPUs
        gpu.updateJPerturb(gamma) #overwrite J front using bi back and J back
    #trialJ = gpus[0].getBuf('J front').read()
    #trialJ = Jbias(trialJ, 0.04)

    for gpu in gpus:
        #gpu.setBuf('J front', trialJ)
        gpu.swapBuf('J') #temporarily put trial J in back buffer
        gpu.perturbMarg() #overwrites bi front using J back
        gpu.swapBuf('J')
    #at this point, front = trial param, back = last accepted param
    
    #read out result and update bimarg
    res = readGPUbufs(['bi front', 'neff', 'weights'], gpus)
    bimargb, Neffs, weightb = res
    Neff = sum(Neffs)
    bimarg_model = sumarr([N*buf for N,buf in zip(Neffs, bimargb)])/Neff
    weights = concatenate(weightb)
    ssr = sum((bimarg_model.flatten() - bimarg_target.flatten())**2)
    trialJ = gpus[0].getBuf('J front').read()
    
    #display result
    print ""
    print ("{}  ssr: {}  Neff: {:.1f} wspan: {:.3g}:{:.3g}").format(
           n, ssr, Neff, min(weights), max(weights))
    print "    trialJ:", printsome(trialJ)
    print "    bimarg:", printsome(bimarg_model)
    print "   weights:", printsome(weights)

    if isinf(Neff) or Neff == 0:
        raise Exception("Error: Divergence. Decrease gamma or increase pc-damp")

    for gpu in gpus:
        gpu.logProfile()

    #check if we accept or reject step
    if ssr > lastssr: 
        return 'rejected', lastssr
    else: 
        #keep this step, and store current J and bm to back buffer
        for gpu in gpus:
            gpu.storeBuf('J') #copy trial J to back buffer
            gpu.setBuf('bi front', bimarg_model)
            gpu.storeBuf('bi')

        #if Neff < nsamples*nwalkers/2 or max(weights) > 64:
        #    return 'finished', ssr

        return 'accepted', ssr

def localDescent((niter, nrepeats), gamma0, gpus):
    gamma = gamma0

    #setup
    for gpu in gpus:
        gpu.calcEnergies('large', 'main')
        gpu.copyBuf('J main', 'J back')
        gpu.copyBuf('J main', 'J front')
        gpu.copyBuf('bi main', 'bi back')
    
    for i in range(nrepeats):
        outJ, outbi = localIter(niter, gamma, gpus)
        for gpu in gpus:
            gpu.swapBuf('J') 
            gpu.storeBuf('J') 
            gpu.swapBuf('bi') 
            gpu.storeBuf('bi') 
    return outJ, outbi

def localIter(niter, gamma, gpus):
    gammasteps = 16
    
    if regularizationScale != 0:
        print "Biasing J down with scale {}".format(regularizationScale)
        biasedJ = Jbias(gpus[0].getBuf('J front').read(), regularizationScale)
        for gpu in gpus:
            gpu.setBuf('J back', biasedJ)

        localStep('bias', 0, inf, gpus)
    
    print "Local target: ", printsome(bimarg_target)
    print "Local optimization:"
    n = 1
    lastssr = inf
    nrepeats = 0
    for i in range(niter): 
        if i != 0 and i % gammasteps == 0:
            if nrepeats == gammasteps:
                print "Too many ssr increases. Stopping local fit."
                break
            gamma = gamma*2
            print "Increasing gamma to {}".format(gamma)
            nrepeats = 0

        result, lastssr = localStep(n, gamma, lastssr, gpus)
        if result == 'accepted':
            n += 1
        if result == 'rejected':
            gamma = gamma/2
            print "Reducing gamma to {} and repeating step".format(gamma)
            nrepeats += 1
        elif result == 'finished':
            print "Sequence weights diverging. Stopping local fit."
            return (gpus[0].getBuf('J front').read(), 
                    gpus[0].getBuf('bi back').read())

    #return back buffer, which contains last accepted move
    return gpus[0].getBuf('J back').read(), gpus[0].getBuf('bi back').read()

################################################################################

def MCMCbenchmark(startseq, couplings, gpus):
    print ""
    print "Benchmarking MCMC for {} loops".format(nloop)
    import time
    
    #initialize
    for gpu in gpus:
        gpu.resetSeqs(startseq)
        gpu.setBuf('J main', couplings)
    
    #warmup
    print "Warmup run..."
    for gpu in gpus:
        gpu.calcEnergies('small', 'main')
    for i in range(nloop):
        for gpu in gpus:
            gpu.runMCMC()
    for gpu in gpus:
        gpu.wait()
    
    #timed run
    print "Timed run..."
    start = time.clock()
    for i in range(nloop):
        for gpu in gpus:
            gpu.runMCMC()
    for gpu in gpus:
        gpu.wait()
    end = time.clock()

    print "Elapsed time: ", end - start, 
    print "Time per loop: ", (end - start)/nloop
    steps_per_second = (nwalkers*nloop*nsteps*L)/(end-start)
    print "MC steps per second: {:g}".format(steps_per_second)

################################################################################
#Run it!!!

print ""
print "MCMC Run"
print "========"

#pre-optimization steps
if args.prestart != 'none':
    pnseq = nsamples*nwalkers/len(gpudevices)
    if args.prestart == 'zero': 
        print "Pre-optimization (random sequences)"
        print "Generating sequences..."
        for gpu in gpus:
            seqs = numpy.random.randint(0,nB,size=(pnseq, L)).astype('<u1')
            gpu.setBuf('seq large', seqs)
        startseq = seqs[0]
    elif args.prestart == 'logscore': 
        print "Pre-optimization (logscore independent sequences)"
        print "Generating sequences..."
        cumprob = cumsum(marg, axis=1)
        cumprob = cumprob/(cumprob[:,-1][:,newaxis]) #correct fp errors?
        for gpu in gpus:
            seqs = array([searchsorted(cp, rand(pnseq)) 
                          for cp in cumprob], dtype='<u1').T.astype('<u1')
            gpu.setBuf('seq large', seqs)
        startseq = seqs[0]
    else:
        #Warning: input couplings should have generated input 
        #"sequences, or weird results ensue
        print "Pre-optimization (loading sequences from dir {})\n".format(
                                                                  args.prestart)
        print "Loading sequences..."
        for n,gpu in enumerate(gpus):
            seqfile = os.path.join(args.prestart, 'seqs-{}'.format(n))
            try:
                seqs = seqload.loadSeqs(seqfile, names=alpha)[0].astype('<u1')
            except:
                raise Exception("Error loading file {}".format(seqfile))
            if seqs.shape[0] != pnseq:
                raise Exception(("Error: Need {} restart sequences, "
                             "got {}").format(pnseq, seqs.shape[0]))
            gpu.setBuf('seq large', seqs)
    
    print "Processing sequences..."
    for gpu in gpus:
        gpu.setBuf('J main', couplings)
        gpu.calcEnergies('large', 'main')
        gpu.calcBimarg('large')
    res = readGPUbufs(['bi main', 'bicount', 'seq large'], gpus)
    bimarg, bicount, seqs = meanarr(res[0]), sumarr(res[1]), res[2]
    
    #store initial setup
    mkdir_p(os.path.join(outdir, 'preopt'))
    print "Unweighted Marginals: ", printsome(bimarg)
    save(os.path.join(outdir, 'preopt', 'initbimarg'), bimarg)
    save(os.path.join(outdir, 'preopt', 'initBicont'), bicount)
    for n,s in enumerate(seqs):
        seqload.writeSeqs(os.path.join(outdir, 'preopt', 'seqs-'+str(n)), 
                          s, alpha)

    rmsd = sqrt(mean((bimarg_target - bimarg)**2))
    ssr = sum((bimarg_target - bimarg)**2)
    wdf = sum(bimarg_target*abs(bimarg_target - bimarg))
    print "S RMSD: {: 9.7f}  SSR: {: 9.5f}  wDf: {: 9.5f}".format(rmsd,ssr,wdf)

    #modify couplings a little
    couplings, bimarg_p = localDescent(perturbSteps, gamma0, gpus)
    save(os.path.join(outdir, 'preopt', 'perturbedbimarg'), bimarg_p)
    save(os.path.join(outdir, 'preopt', 'perturbedJ'), couplings)
else:
    print "No Pre-optimization"

if args.benchmark:
    MCMCbenchmark(startseq, couplings, gpus)
elif args.measureFPerror:
    gpus[0].measureFPerror()
else:
    doFit(startseq, couplings, gpus)

print "Done!"


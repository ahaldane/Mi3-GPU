#!/usr/bin/env python2
from __future__ import print_function
from scipy import *
from scipy.misc import logsumexp
import scipy
import numpy as np
from numpy.random import randint
import pyopencl as cl
import pyopencl.array as cl_array
import sys, os, errno, argparse, time, ConfigParser
import seqload
from changeGauge import zeroGauge, zeroJGauge, fieldlessGaugeEven
from mcmcGPU import setupGPUs, initGPU, divideWalkers, printGPUs, readGPUbufs
from NewtonSteps import newtonMCMC, runMCMC

################################################################################
# Set up enviroment and some helper functions

progname = 'IvoGPU.py'

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc: 
        if not (exc.errno == errno.EEXIST and os.path.isdir(path)):
            raise
scriptPath = os.path.dirname(os.path.realpath(__file__))
printsome = lambda a: " ".join(map(str,a.flatten()[-5:]))

class attrdict(dict):
    def __getattr__(self, attr):
        try:
            return dict.__getitem__(self, attr)
        except KeyError:
            return None

def seqsize_from_param_shape(shape):
    L = int(((1+sqrt(1+8*shape[0]))/2) + 0.5) 
    nB = int(sqrt(shape[1]) + 0.5) 
    return L, nB

def getUnimarg(bimarg):
    L, nB = seqsize_from_param_shape(bimarg.shape)
    ff = bimarg.reshape((L*(L-1)/2,nB,nB))
    f = array([sum(ff[0],axis=1)] + [sum(ff[n],axis=0) for n in range(L-1)])
    return f/(sum(f,axis=1)[:,newaxis]) # correct any fp errors

#identical calculation as CL kernel, but with high precision (to check fp error)
def getEnergiesMultiPrec(s, couplings): 
    from mpmath import mpf, mp
    mp.dps = 32
    couplings = [[mpf(float(x)) for x in r] for r in couplings]
    pairenergy = [mpf(0) for n in range(s.shape[0])]
    for n,(i,j) in enumerate([(i,j) for i in range(L-1) for j in range(i+1,L)]):
        r = couplings[n]
        cpl = (r[b] for b in (nB*s[:,i] + s[:,j]))
        pairenergy = [x+n for x,n in zip(pairenergy, cpl)]
    return pairenergy

################################################################################

def optionRegistry():
    options = []
    add = lambda opt, **kwds: options.append((opt, kwds))

    # option used by both potts and sequence loaders, designed
    # to load in the output of a previous run
    add('seqmodel', default=None,
        help=("One of 'zero', 'logscore', or a directory name. Generates or "
              "loads 'alpha', 'couplings', 'startseq' and 'seqs', if not "
              "otherwise supplied.") )
    add('outdir', default='output', help='Output Directory')

    # GPU options
    add('nwalkers', type=uint32,
        help="Number of MC walkers")
    add('nsteps', type=uint32, default=2048,
        help="number of MC steps per kernel call")
    add('wgsize', type=int, default=256, 
        help="GPU workgroup size")
    add('gpus', 
        help="GPUs to use (comma-sep list of platforms #s, eg '0,0')")
    add('profile', action='store_true', 
        help="enable OpenCL profiling")
    add('nlargebuf', type=uint32, default=1,
        help='size of large seq buffer, in multiples of nwalkers')
    add('measurefperror', action='store_true', 
        help="enable fp error calculation")
    add('gibbs', action='store_true',
        help='Use gibbs sampling instead of metropoils-hastings')
    
    # Newton options
    add('bimarg', required=True,
        help="Target bivariate marginals (npy file)")
    add('mcsteps', type=uint32, required=True, 
        help="Number of rounds of MCMC generation") 
    add('newtonsteps', default='128', type=uint32, 
        help="Number of newton steps per round.")
    add('gamma', type=float32, required=True, 
        help="Initial step size")
    add('damping', default=0.001, type=float32, 
        help="Damping parameter")
    add('jclamp', default=0, type=float32, 
        help="Clamps maximum change in couplings per newton step")
    add('preopt', action='store_true', 
        help="Perform a round of newton steps before first MCMC run") 
    add('resetseqs', action='store_false', 
        help="Reset sequence to S0 at start of every MCMC round") 
    
    # Potts options
    add('alpha', required=True,
        help="Alphabet, a sequence of letters")
    add('couplings', 
        help="One of 'zero', 'logscore', or a filename")
    add('L', help="sequence length") 
    
    # Sequence options
    add('startseq', help="Starting sequence. May be 'rand'") 
    add('seqs', help="File containing sequences to pre-load to GPU") 

    # Sampling Param
    add('equiltime', type=uint32, required=True, 
        help="Number of MC kernel calls to equilibrate")
    add('sampletime', type=uint32, required=True, 
        help="Number of MC kernel calls between samples")
    add('nsamples', type=uint32, required=True, 
        help="Number of sequence samples")
    add('trackequil', type=uint32, default=0,
        help='Save bimarg every TRACKEQUIL steps during equilibration')

    return dict(options)

def addopt(parser, groupname, optstring):
    if groupname is not None:
        group = parser.add_argument_group(groupname)
        add = group.add_argument
    else:
        add = parser.add_argument

    for option in optstring.split():
        optargs = addopt.options[option]
        add('--' + option, **optargs)
addopt.options = optionRegistry()

def requireargs(args, required):
    required = required.split()
    args = vars(args)
    for r in required:
        if args[r] is None:
            raise Exception("error: argument --{} is required".format(r))

################################################################################

def inverseIsing(args, log):
    descr = ('Inverse Ising inference using a quasi-Newton MCMC algorithm '
             'on the GPU')
    parser = argparse.ArgumentParser(prog=progname + ' inverseIsing',
                                     description=descr)
    addopt(parser, 'GPU options',         'nwalkers nsteps wgsize '
                                          'gibbs gpus profile')
    addopt(parser, 'Sequence Options',    'startseq seqs')
    addopt(parser, 'Newton Step Options', 'bimarg mcsteps newtonsteps gamma '
                                          'damping jclamp preopt resetseqs')
    addopt(parser, 'Sampling Options',    'equiltime sampletime nsamples '
                                          'trackequil')
    addopt(parser, 'Potts Model Options', 'alpha couplings L')
    addopt(parser,  None,                 'seqmodel outdir')

    args = parser.parse_args(args)
    args.nlargebuf = args.nsamples
    args.measurefperror = False

    log("Initialization")
    log("===============")
    log("")
    
    p = attrdict({'outdir': args.outdir})
    mkdir_p(args.outdir)

    p.update(process_newton_args(args, log))
    if p.bimarg is not None:
        p['L'], p['nB'] = seqsize_from_param_shape(p.bimarg.shape)

    p.update(process_potts_args(args, p.L, p.nB, p.bimarg, log))
    L, nB, alpha = p.L, p.nB, p.alpha

    p.update(process_sample_args(args, log))
    rngPeriod = (p.equiltime + p.sampletime*p.nsamples)*p.mcmcsteps
    gpup, cldat, gdevs = process_GPU_args(args, L, nB, p.outdir, rngPeriod, log)
    p.update(gpup)
    gpuwalkers = divideWalkers(p.nwalkers, len(gdevs), p.wgsize, log)
    gpus = [initGPU(n, cldat, dev, nwalk, nwalk*p.nsamples, p, log)
            for n,(dev, nwalk) in enumerate(zip(gdevs, gpuwalkers))]
    log("")
    #log(("Running {} MCMC walkers in parallel over {} GPUs, with {} MC "
    #    "steps per kernel call").format(p.nwalkers, len(gpus), 
    #    p.nsteps))

    preopt_seqs = sum([g.nseq['large'] for g in gpus])
    p.update(process_sequence_args(args, L, alpha, p.bimarg, log, 
                                   nseqs=preopt_seqs))
    if p.preopt:
        if p.seqs is None:
            raise Exception("Need to provide seqs if using pre-optimization")
        transferSeqsToGPU(gpus, 'large', p.seqs, log)
    
    #if we're not initializing seqs to single sequence, need to load
    # a set of initial sequences into small buffer
    if not p.resetseqs:
        if p.seqs is None:
            raise Exception("Need to provide seqs if not using startseq")
        #get required seqs from end of detected seqs
        if len(p.seqs) == 1:
            smallseq = [p.seqs[-sum([g.nseq['small'] for f in gpus]):]]
        else:
            smallseq = [s[-g.nseq['small']:] for s,g in zip(p.seqs, gpus)]
        transferSeqsToGPU(gpus, 'small', smallseq, log)
    log("")

    log("Computation Overview")
    log("====================")
    log("Running {} Newton-MCMC rounds, with {} parameter update steps per "
        "round.".format(p.mcmcsteps, p.newtonSteps))
    log(("In each round, running {} MC walkers for {} equilibration loops then "
         "sampling every {} loops to get {} samples ({} total seqs) with {} MC "
         "steps per loop (Each walker equilibrated a total of {} MC steps)."
         ).format(p.nwalkers, p.equiltime, p.sampletime, p.nsamples, 
                p.nsamples*p.nwalkers, p.nsteps, p.nsteps*p.equiltime))
    log("")
    log("")
    log("MCMC Run")
    log("========")

    newtonMCMC(p, gpus, log)

def getEnergies(args, log):
    descr = ('Compute Potts Energy of a set of sequences')
    parser = argparse.ArgumentParser(prog=progname + ' getEnergies',
                                     description=descr)
    add = parser.add_argument
    add('out', default='output', help='Output File')
    addopt(parser, 'GPU Options',         'wgsize gpus profile')
    addopt(parser, 'Potts Model Options', 'alpha couplings')
    addopt(parser, 'Sequence Options',    'seqs')
    addopt(parser,  None,                 'outdir')

    #genenergies uses a subset of the full inverse ising parameters,
    #so use custom set of params here
    
    args = parser.parse_args(args)
    args.measurefperror = False

    requireargs(args, 'couplings alpha seqs')

    log("Initialization")
    log("===============")
    log("")

    param = attrdict({'outdir': args.outdir})
    param.update(process_potts_args(args, None, None, None, log))
    L, nB, alpha = param.L, param.nB, param.alpha
    log("Sequence Setup")
    log("--------------")
    seqs = loadSequenceFile(args.seqs, alpha, log)
    if seqs is None:
        raise Exception("seqs must be supplied")
    log("")

    args.nwalkers = len(seqs)
    args.gibbs = False
    args.nsteps = 1
    args.nlargebuf = 1
    gpup, cldat, gdevs = process_GPU_args(args, L, nB, p.outdir, 1, log)
    gpuwalkers = divideWalkers(p.nwalkers, len(gdevs), p.wgsize, log)
    param.update(gpup)
    gpus = [initGPU(n, cldat, dev, nwalk, 1, param, log)
            for n,(dev, nwalk) in enumerate(zip(gdevs, gpuwalkers))]
    transferSeqsToGPU(gpus, 'small', [seqs], log)
    log("")


    log("Computing Energies")
    log("==================")
    
    for gpu in gpus:
        gpu.setBuf('J main', param.couplings)
    
    for gpu in gpus:
        gpu.calcEnergies('small', 'main')
    es = concatenate(readGPUbufs(['E small'], gpus)[0])
    
    log("Saving results to file '{}'".format(args.out))
    save(args.out, es)


def MCMCbenchmark(args, log):
    descr = ('Benchmark MCMC generation on the GPU')
    parser = argparse.ArgumentParser(prog=progname + ' benchmark',
                                     description=descr)
    add = parser.add_argument
    add('--nloop', type=uint32, required=True, 
        help="Number of kernel calls to benchmark")
    addopt(parser, 'GPU options',         'nwalkers nsteps wgsize '
                                          'gibbs gpus profile')
    addopt(parser, 'Sequence Options',    'startseq seqs')
    addopt(parser, 'Potts Model Options', 'alpha couplings L')
    addopt(parser,  None,                 'seqmodel outdir')

    args = parser.parse_args(args)
    nloop = args.nloop
    args.measurefperror = False

    log("Initialization")
    log("===============")
    log("")

    p = attrdict({'outdir': args.outdir})
    mkdir_p(args.outdir)
    p.update(process_potts_args(args, p.L, p.nB, None, log))
    L, nB, alpha = p.L, p.nB, p.alpha
    args.nlargebuf = 1
    gpup, cldat, gdevs = process_GPU_args(args, L, nB, p.outdir, 2*nloop, log)
    p.update(gpup)
    gpuwalkers = divideWalkers(p.nwalkers, len(gdevs), p.wgsize, log)
    gpus = [initGPU(n, cldat, dev, nwalk, 1, p, log)
            for n,(dev, nwalk) in enumerate(zip(gdevs, gpuwalkers))]
    log("")
    preopt_seqs = sum([g.nseq['small'] for g in gpus])
    p.update(process_sequence_args(args, L, alpha, None, log,nseqs=preopt_seqs))
    
    if p.seqs is not None:
        transferSeqsToGPU(gpus, 'small', p.seqs, log)
    elif p.startseq is not None:
        log("Loading small seq buffer with startseq")
        for gpu in gpus:
            gpu.fillSeqs(p.startseq)
    else:
        raise Exception("Error: To benchmark, must either supply startseq or "
                        "load seqs into small seq buffer")
    log("")


    log("Benchmark")
    log("=========")
    log("")
    log("Benchmarking MCMC for {} loops, {} MC steps per loop".format(
                                                 nloop, p.nsteps))
    import time

    def runMCMC():
        for i in range(nloop):
            for gpu in gpus:
                gpu.runMCMC()
        for gpu in gpus:
            gpu.wait()
    
    #initialize
    for gpu in gpus:
        gpu.setBuf('J main', p.couplings)
    
    #warmup
    log("Warmup run...")
    runMCMC()
    
    #timed run
    log("Timed run...")
    start = time.clock()
    runMCMC()
    end = time.clock()

    log("Elapsed time: ", end - start, )
    totsteps = p.nwalkers*nloop*p.nsteps
    steps_per_second = totsteps/(end-start)
    log("MC steps computed: {}".format(totsteps))
    log("MC steps per second: {:g}".format(steps_per_second))

def equilibrate(args, log):
    descr = ('Run a round of MCMC generation on the GPU')
    parser = argparse.ArgumentParser(prog=progname + ' mcmc',
                                     description=descr)
    add = parser.add_argument
    addopt(parser, 'GPU options',         'nwalkers nsteps wgsize '
                                          'gibbs gpus profile')
    addopt(parser, 'Sequence Options',    'startseq seqs')
    addopt(parser, 'Sampling Options',    'equiltime sampletime nsamples '
                                          'trackequil')
    addopt(parser, 'Potts Model Options', 'alpha couplings L')
    addopt(parser,  None,                 'seqmodel outdir')

    args = parser.parse_args(args)
    args.measurefperror = False

    log("Initialization")
    log("===============")

    p = attrdict({'outdir': args.outdir})
    mkdir_p(args.outdir)

    p.update(process_potts_args(args, None, None, None, log))
    L, nB, alpha = p.L, p.nB, p.alpha

    p.update(process_sample_args(args, log))
    rngPeriod = (p.equiltime + p.sampletime*p.nsamples)
    gpup, cldat, gdevs = process_GPU_args(args, L, nB, p.outdir, rngPeriod, log)
    p.update(gpup)
    gpuwalkers = divideWalkers(p.nwalkers, len(gdevs), p.wgsize, log)
    gpus = [initGPU(n, cldat, dev, nwalk, nwalk*p.nsamples, p, log)
            for n,(dev, nwalk) in enumerate(zip(gdevs, gpuwalkers))]
    preopt_seqs = sum([g.nseq['large'] for g in gpus])
    p.update(process_sequence_args(args, L, alpha, None, log, 
                                   nseqs=preopt_seqs))
    log("")

    log("Equilibrating ...")
    for gpu in gpus:
        gpu.fillSeqs(p.startseq)

    (bimarg_model, 
     bicount, 
     energies, 
     seqs) = runMCMC(gpus, p.startseq, p.couplings, '.', p)
    
    outdir = p.outdir
    savetxt(os.path.join(outdir, 'bicounts'), bicount, fmt='%d')
    save(os.path.join(outdir, 'bimarg'), bimarg_model)
    save(os.path.join(outdir, 'energies'), energies)
    for n,seqbuf in enumerate(seqs):
        seqload.writeSeqs(os.path.join(outdir, 'seqs-{}'.format(n)), 
                          seqbuf, alpha)


def subseqFreq(args, log):
    descr = ('Compute relative frequency of subsequences at fixed positions')
    parser = argparse.ArgumentParser(prog=progname + ' subseqFreq',
                                     description=descr)
    add = parser.add_argument
    add('fixpos', help="comma separated list of fixed positions")
    add('out', default='output', help='Output File')
    addopt(parser, 'GPU options',         'nsteps wgsize gpus profile')
    addopt(parser, 'Potts Model Options', 'alpha couplings L')
    addopt(parser,  None,                 'outdir')
    group = parser.add_argument_group('Sequence Options')
    add = group.add_argument
    add('backgroundseqs')
    add('subseqs')
    
    args = parser.parse_args(args)
    args.measurefperror = False

    log("Initialization")
    log("===============")
    log("")
    
    p = attrdict({'outdir': args.outdir})
    args.trackequil = 0
    mkdir_p(args.outdir)
    p.update(process_potts_args(args, p.L, p.nB, None, log))
    L, nB, alpha = p.L, p.nB, p.alpha

    # try to load sequence files
    bseqs = loadSequenceFile(args.backgroundseqs, alpha, log)
    sseqs = loadSequenceFile(args.subseqs, alpha, log)

    gpup, cldat, gdevs = process_GPU_args(args, L, nB, p.outdir, 1, log)
    p.update(gpup)
    gpuwalkers = divideWalkers(len(bseqs), len(gdevs), p.wgsize, log)
    gpus = [initGPU(n, cldat, dev, len(sseqs), nwalk, p, log)
            for n,(dev, nwalk) in enumerate(zip(gdevs, gpuwalkers))]

    #fix positions
    fixedpos = array([int(x) for x in args.fixpos.split(',')])
    fixedmarks = zeros(L, dtype='u1')
    fixedmarks[fixedpos] = 1
    
    #load buffers
    gpubseqs = split(bseqs, cumsum(gpuwalkers)[:-1])
    for gpu,bs in zip(gpus, gpubseqs):
        gpu.setBuf('seq small', sseqs)
        gpu.setBuf('seq large', bs)
        gpu.setBuf('fixpos', fixedmarks)
        gpu.setBuf('J main', p.couplings)
    log("")

    log("Subsequence Frequency Calculation")
    log("=================================")
    log("")

    for gpu in gpus:
        gpu.calcEnergies('large', 'main')
    origEs = concatenate(readGPUbufs(['E large'], gpus)[0])

    log("Getting substituted energies...")
    subseqE = []
    logf = zeros(len(sseqs))
    for n in range(len(sseqs)):
        # replaced fixed positions by subsequence, and calc energies
        for gpu in gpus:
            gpu.copySubseq(n)
            gpu.calcEnergies('large', 'main')
        energies = concatenate(readGPUbufs(['E large'], gpus)[0])
        logf[n] = logsumexp(origEs - energies)

    #save result
    log("Saving result (log frequency) to file {}".format(args.out))
    save(args.out, logf)
    

################################################################################

def process_GPU_args(args, L, nB, outdir, rngPeriod, log):
    log("GPU setup")
    log("---------")

    param = attrdict({'nsteps': args.nsteps,
                      'wgsize': args.wgsize,
                      'nwalkers': args.nwalkers,
                      'gpuspec': args.gpus,
                      'gibbs': args.gibbs,
                      'profile': args.profile,
                      'fperror': args.measurefperror})
    
    p = attrdict(param.copy())
    p.update({'L': L, 'nB': nB, 'outdir': outdir, 'rngPeriod': rngPeriod})
    
    scriptfile = os.path.join(scriptPath, "mcmc.cl")

    log("Work Group Size: {}".format(p.wgsize))
    log("{} MC steps per MCMC kernel call".format(p.nsteps))
    log("Using {} MC sampler".format('Gibbs' if args.gibbs 
                                   else 'Metropolis-hastings'))
    log("GPU Initialization:")
    if p.profile:
        log("Profiling Enabled")
    clinfo, gpudevs = setupGPUs(scriptPath, scriptfile, p, log)

    log("")
    return p, clinfo, gpudevs

def process_newton_args(args, log):
    log("Newton Solver Setup")
    log("-------------------")
    mcmcsteps = args.mcsteps  
    log("Running {} Newton-MCMC rounds".format(mcmcsteps))

    param = {'mcmcsteps': args.mcsteps,
             'newtonSteps': args.newtonsteps,
             'gamma0': args.gamma,
             'pcdamping': args.damping,
             'jclamp': args.jclamp,
             'resetseqs': args.resetseqs,
             'preopt': args.preopt }
    p = attrdict(param)

    cutoffstr = ('dJ clamp {}'.format(p.jclamp) if p.jclamp != 0 
                 else 'no dJ clamp')
    log(("Updating J locally with gamma = {}, {}, and pc-damping {}. "
         "Running {} Newton update steps per round.").format(
          p.gamma0, cutoffstr, p.pcdamping, p.newtonSteps))

    log("Reading target marginals from file {}".format(args.bimarg))
    bimarg = scipy.load(args.bimarg)
    if bimarg.dtype != dtype('<f4'):
        raise Exception("Bimarg in wrong format")
        #could convert, but this helps warn that something may be wrong
    if any(~((bimarg.flatten() >= 0) & (bimarg.flatten() <= 1))):
        raise Exception("Bimarg must be nonzero and 0 < f < 1")
    log("Target Marginals: " + printsome(bimarg) + "...")

    log("")
    p['bimarg'] = bimarg
    return p

def updateLnB(L, nB, newL, newnB, name):
    # update L and nB with new values, checking that they
    # are the same as the old values if not None
    if newL is not None:
        if L is not None and L != newL:
            raise Exception("L from {} ({}) inconsitent with previous "
                            "value ({})".format(name, newL, L))
        L = newL
    if newnB is not None:
        if nB is not None and nB != newnB:
            raise Exception("nB from {} ({}) inconsitent with previous "
                            "value ({})".format(name, newnB, nB))
        nB = newnB
    return L, nB

def process_potts_args(args, L, nB, bimarg, log):
    log("Potts Model Setup")
    log("-----------------")

    # we try to infer L and nB from any values given. The possible sources
    # * command line options -L and -nB
    # * from bivariate_target dimensions
    # * from coupling dimensions
    
    alpha = args.alpha
    argL = args.L if 'L' in args else None
    L, nB = updateLnB(argL, len(alpha), L, nB, 'bimarg')
    
    # next try to get couplings (may determine L, nB)
    couplings, L, nB = getCouplings(args, L, nB, bimarg, log)
    # we should have L and nB by this point

    log("alphabet: {}".format(alpha))
    log("nBases {}  seqLen {}".format(nB, L))
    log("Couplings: " + printsome(couplings) + "...")

    log("")
    return attrdict({'L': L, 'nB': nB, 'alpha': alpha, 
                     'couplings': couplings})

def getCouplings(args, L, nB, bimarg, log):
    couplings = None

    if args.seqmodel and args.seqmodel in ['zero', 'logscore']:
        args.couplings = args.seqmodel

    if args.couplings is not None:
        #first try to generate couplings (requires L, nB)
        if args.couplings in ['zero', 'logscore']:
            if L is None: # we are sure to have nB
                raise Exception("Need L to generate couplings")
        if args.couplings == 'zero':
            log("Setting Initial couplings to 0")
            couplings = zeros((L*(L-1)/2, nB*nB), dtype='<f4')
        elif args.couplings == 'logscore':
            log("Setting Initial couplings to Independent Log Scores")
            if bimarg is None:
                raise Exception("Need bivariate marginals to generate "
                                "logscore couplings")
            h = -np.log(getUnimarg(bimarg))
            J = zeros((L*(L-1)/2,nB*nB), dtype='<f4')
            couplings = fieldlessGaugeEven(h, J)[1]
        else: #otherwise load them from file
            log("Reading couplings from file {}".format(args.couplings))
            couplings = scipy.load(args.couplings)
            if couplings.dtype != dtype('<f4'):
                raise Exception("Couplings must be in 'f4' format")
    elif args.seqmodel and args.seqmodel not in ['zero', 'logscore']:
        # and otherwise try to load them from model directory
        fn = os.path.join(args.seqmodel, 'J.npy')
        if os.path.isfile(fn):
            log("Reading couplings from file {}".format(fn))
            couplings = scipy.load(fn)
            if couplings.dtype != dtype('<f4'):
                raise Exception("Couplings must be in 'f4' format")
    L2, nB2 = seqsize_from_param_shape(couplings.shape)
    L, nB = updateLnB(L, nB, L2, nB2, 'couplings')

    if couplings is None:
        raise Exception("Could not find couplings. Use either the "
                        "'couplings' or 'seqmodel' options.")

    return couplings, L, nB

def process_sequence_args(args, L, alpha, bimarg, log, nseqs=None):
    log("Sequence Setup")
    log("--------------")

    nB = len(alpha)
    startseq, seqs = None, None

    # check if we were asked to generate sequences
    if any([arg in ['zero', 'logscore'] for arg in [args.seqmodel, args.seqs]]):
        if args.seqs is not None and args.seqmodel is not None:
            raise Exception("Cannot specify both seqs and "
                            "seqmodel=[rand, logscore]")
        if nseqs is None:
            raise Exception("Cannot generate sequences without known nseq")
        seqs = [generateSequences(args.seqmodel, L, nB, nseqs, bimarg, log)]
        startseq = seqs[0][0]
        startseq_origin = 'generated ' + args.seqmodel
        seqmodeldir = None
    else:
        seqmodeldir = args.seqmodel

    # try to load sequence files
    if args.seqs not in [None, 'zero', 'logscore']:
        seqs = [loadSequenceFile(args.seqs, alpha, log)]
    elif seqmodeldir is not None:
        seqs = loadSequenceDir(seqmodeldir, alpha, log)

    # try to get start seq
    if args.startseq:
        if args.startseq == 'rand':
            startseq = randint(0, nB, size=L).astype('<u1')
            startseq_origin = 'random'
        else: # given string
            startseq = array(map(alpha.index, args.startseq), dtype='<u1')
            startseq_origin = 'supplied'
    elif seqmodeldir is not None:
        fn = os.path.join(seqmodeldir, 'startseq')
        if os.path.isfile(fn):
            log("Reading startseq from file {}".format(fn))
            with open(fn) as f:
                startseq = f.readline().strip()
                startseq = array(map(alpha.index, startseq), dtype='<u1')
            startseq_origin = 'from file'

    if seqs is None:
        log("No sequence dataset loaded")

    if startseq is not None:
        log("Start seq ({}): {}".format(startseq_origin, 
                                       "".join(alpha[x] for x in startseq)))
    else:
        log("No start seq supplied")

    log("")
    return attrdict({'startseq': startseq, 'seqs': seqs})

def generateSequences(gentype, L, nB, nseqs, bimarg, log):
    if gentype == 'zero': 
        log("Generating {} random sequences...".format(nseqs))
        return randint(0,nB,size=(nseqs, L)).astype('<u1')
    elif gentype == 'logscore': 
        log("Generating {} logscore-independent sequences...".format(nseqs))
        if bimarg is None:
            raise Exception("Bimarg must be provided to generate sequences")
        ff = bimarg.reshape((L*(L-1)/2,nB,nB))
        marg = (array([sum(ff[0],axis=1)] + 
                [sum(ff[n],axis=0) for n in range(L-1)]))
        marg = marg/(sum(marg,axis=1)[:,newaxis]) # correct any fp errors
        cumprob = cumsum(marg, axis=1)
        cumprob = cumprob/(cumprob[:,-1][:,newaxis]) #correct fp errors?
        return array([searchsorted(cp, rand(nseqs)) for cp in cumprob], 
                     dtype='<u1').T

def loadSequenceFile(sfile, alpha, log):
    log("Loading sequences from file {}".format(sfile))
    seqs = seqload.loadSeqs(sfile, names=alpha)[0].astype('<u1')
    return seqs

def loadSequenceDir(sdir, alpha, log):
    log("Loading sequences from dir {}".format(sdir))
    seqs = []
    while True:
        sfile = os.path.join(sdir, 'seqs-{}'.format(len(seqs)))
        if not os.path.exists(sfile):
            break
        seqs.append(seqload.loadSeqs(sfile, names=alpha)[0].astype('<u1'))
    return seqs

def transferSeqsToGPU(gpus, bufname, seqs, log):
    log("Transferring {} seqs to gpu's {} seq buffer...".format(str([len(s) for s in seqs]), bufname))
    if len(seqs) == 1:
        # split up seqs into parts for each gpu
        seqs = seqs[0]
        sizes = [g.nseq[bufname] for g in gpus]

        if len(seqs) == sum(sizes):
            seqs = split(seqs, cumsum(sizes)[:-1])
        else:
            raise Exception(("Expected {} total sequences, got {}").format(
                             sum(sizes), len(seqs)))

    for n,(gpu,seq) in enumerate(zip(gpus, seqs)):
        if len(seq) != gpu.nseq[bufname]:
            raise Exception("Expected {} sequences, got {}".format(
                            gpu.nseq[bufname], len(seq)))
        gpu.setBuf('seq ' + bufname, seq)

def process_sample_args(args, log):
    p = attrdict({'equiltime': args.equiltime,
                  'sampletime': args.sampletime,
                  'nsamples': args.nsamples,
                  'trackequil': args.trackequil})

    if p.nsamples == 0:
        raise Exception("nsamples must be at least 1")

    log("MCMC Sampling Setup")
    log("-------------------")
    log(("In each MCMC round, running {} GPU MCMC kernel calls then sampling "
         "every {} kernel calls to get {} samples").format(p.equiltime, 
                                             p.sampletime, p.nsamples))

    if p.trackequil != 0:
        if p.equiltime%p.trackequil != 0:
            raise Exception("Error: trackequil must be a divisor of equiltime")
        log("Tracking equilibration every {} loops.".format(p.trackequil))

    log("")
    return p

################################################################################

class CLInfoAction(argparse.Action):
    def __init__(self, option_strings, dest=argparse.SUPPRESS, 
                 default=argparse.SUPPRESS, help=None):
        super(CLInfoAction, self).__init__(option_strings=option_strings,
            dest=dest, default=default, nargs=0, help=help)
    def __call__(self, parser, namespace, values, option_string=None):
        printGPUs(print)
        parser.exit()

def readConfig(fp, section):
    config = ConfigParser.SafeConfigParser()
    config.readfp(fp)
    sections = config.sections()
    if len(sections) != 1 or sections[0] != section:
        raise Exception("Config input must have only one section with the "
                        "same name as the specified actions")
    return config.items(section)

def main(args):
    actions = {
      'inverseIsing':   inverseIsing,
      'getEnergies':    getEnergies,
      'benchmark':      MCMCbenchmark,
      #'measureFPerror': measureFPerror,
      'subseqFreq':     subseqFreq,
      'mcmc':            equilibrate
     }
    
    descr = 'Perform biophysical Potts Model calculations on the GPU'
    parser = argparse.ArgumentParser(description=descr, add_help=False)
    add = parser.add_argument
    add('action', choices=actions.keys(), nargs='?', default=None,
        help="Computation to run")
    add('--clinfo', action=CLInfoAction, help="Display detected GPUs")
    add('-h', '--help', action='store_true', 
        help="show this help message and exit")


    known_args, remaining_args = parser.parse_known_args(args)

    if known_args.action is None:
        if known_args.help:
            print(parser.format_help())
            return
        print(parser.format_usage())
        return

    if known_args.help:
        remaining_args.append('-h')
    
    #if not sys.stdin.isatty(): #config file supplied in stdin
    #    config = readConfig(sys.stdin, known_args.action)
    #    configargs = [arg for opt,val in config for arg in ('--'+opt, val)]
    #    remaining_args = configargs + remaining_args

    actions[known_args.action](remaining_args, print)

main(sys.argv[1:])

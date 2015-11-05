#!/usr/bin/env python2
from __future__ import print_function
from scipy import *
import scipy
import numpy as np
from numpy.random import randint
import numpy.random
import pyopencl as cl
import pyopencl.array as cl_array
import sys, os, errno, glob, argparse, time
import ConfigParser
import seqload
from scipy.optimize import leastsq
from changeGauge import zeroGauge, zeroJGauge, fieldlessGaugeEven
from mcmcGPU import initGPUs, printGPUs
from NewtonSteps import newtonMCMC

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

################################################################################

def optionRegistry():
    options = []
    add = lambda opt, **kwds: options.append((opt, kwds))

    # option used by both potts and sequence loaders, designed
    # to load in the output of a previous run
    add('seqmodel', default='none',
        help=("One of 'zero', 'logscore', or a directory name. Generates or "
              "loads 'alpha', 'couplings', 'startseq' and 'seqs', if not "
              "otherwise supplied.") )
    add('outdir', default='output', help='Output Directory')

    # GPU options
    add('nwalkers', type=uint32,
        help="Number of MC walkers")
    add('nsteps', type=uint32, default=16,
        help="number of MC steps per kernel call, in multiples of L")
    add('wgsize', type=int, default=256, 
        help="GPU workgroup size")
    add('gpus', 
        help="GPUs to use (comma-sep list of platforms #s, eg '0,0')")
    add('profile', action='store_true', 
        help="enable OpenCL profiling")
    add('nlargebuf', type=uint32, default=1
        help='size of large seq buffer, in multiples of nwalkers')
    
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

    return options

def addopt(parser, group, optstring):
    if group is not None:
        group = parser.add_group(groupname)
        add = group.add_argument
    else:
        add = parser.add_argument

    for option in optstring.split():
        optname, optargs = addopt.options[option]
        add('--' + optname, **optargs)
addopt.options = optionRegistry()

################################################################################

def inverseIsing(args, log):
    descr = ('Inverse Ising inference using a quasi-Newton MCMC algorithm '
             'on the GPU')
    parser = argparse.ArgumentParser(prog=progname + ' inverseIsing',
                                     description=descr)
    addopt(parser, 'GPU options',         'nwalkers nsteps wgsize gpus profile')
    addopt(parser, 'Sequence Options',    'startseq seqs')
    addopt(parser, 'Newton Step Options', 'bimarg mcsteps newtonsteps gamma '
                                          'damping jclamp preopt')
    addopt(parser, 'Sampling Options',    'equiltime sampletime nsamples '
                                          'trackequil')
    addopt(parser, 'Potts Model Options', 'alpha couplings L')
    addopt(parser,  None,                 'seqmodel outdir')

    add = parser.add_argument
    
    args = parser.parse_args(args)
    args.nlargebuf = args.nsamples

    log("Initialization")
    log("===============")
    log("")
    
    param = attrdict({'outdir': args.outdir})
    mkdir_p(args.outdir)

    param.update(NewtonParam.process_args(args, param, log))
    if param.bimarg is not None:
        param['L'], param['nB'] = seqsize_from_param_shape(param.bimarg.shape)
    param.update(PottsParam.process_args(args, param, log))
    param.update(SampleParam.process_args(args, param, log))
    rngPeriod = param.nloop*param.mcmcsteps
    gpuparam, gpus = GPUParam.process_args(args, param, rngPeriod, log)
    param.update(gpuparam)
    param.update(SeqParam.process_args(args, param, gpus, log))
    log("")

    p = param
    log("Computation Overview")
    log("====================")
    log("Running {} Newton-MCMC rounds, with {} parameter update steps per "
        "round.".format(p.mcmcsteps, p.newtonSteps))
    log(("In each round, running {} MC walkers for {} equilibration loops then "
         "sampling every {} loops to get {} samples ({} total seqs) with {} MC "
         "steps per loop (Each walker equilibrated a total of {} MC steps)."
         ).format(p.nwalkers, p.nloop, p.nsampleloops, p.nsamples, 
                p.nsamples*p.nwalkers, p.nsteps*p.L, p.nsteps*p.L*p.nloop))
    log("")
    log("")
    log("MCMC Run")
    log("========")

    newtonMCMC(param, gpus, log)

def getEnergies(args, log):
    descr = ('Compute Potts Energy of a set of sequences')
    parser = argparse.ArgumentParser(prog=progname + ' getEnergies',
                                     description=descr)
    addopt(parser, 'GPU Options',         'wgsize gpus profile')
    addopt(parser, 'Potts Model Options', 'alpha couplings')
    addopt(parser, 'Sequence Options',    'seqs')

    add = parser.add_argument
    add('out', default='output', help='Output File')

    #genenergies uses a subset of the full inverse ising parameters,
    #so use custom set of params here
    
    args = parser.parse_args(args)
    args.nwalkers = args.nseq
    args.nsteps = 1

    if args.out is None:
        raise Exception("no output file specified") #XXX better error

    param.update(PottsParam.process_args(args, param, log))
    gpuparam, gpus = GPUParam.process_args(args, param, 2*nloop, log)
    param.update(gpuparam)
    param.update(SeqParam.process_args(args, param, gpus, log))
    log("")

    if not param.sseq_l:
        raise Exception("Need to supply sequences")

    for gpu in gpus:
        gpu.setBuf('J main', param.couplings)
    
    for gpu in gpus:
        gpu.calcEnergies('small', 'main')
    es = readGPUbufs(['E small'], gpus)
    
    log("Saving results to file '{}'".format(param.out))
    save(param.out, es)


def MCMCbenchmark(args, log):
    descr = ('Benchmark MCMC generation on the GPU')
    parser = argparse.ArgumentParser(prog=progname + ' benchmark',
                                     description=descr)
    add = parser.add_argument
    add('--nloop', type=uint32, required=True, 
        help="Number of kernel calls to benchmark")
    add('--outdir', default='output')
    add('--seqmodel', default='none',
        help=("One of 'zero', 'logscore', or a directory name. Generates or "
              "loads 'couplings', 'startseq' and 'seqs', if not otherwise "
              "supplied.") )

    log("Initialization")
    log("===============")
    log("")
    
    GPUParam.add_arguments(parser)
    SeqParam.add_arguments(parser)
    PottsParam.add_arguments(parser)

    args = parser.parse_args(args)
    nloop = args.nloop
    args.nlargebuf = 1
    
    param = attrdict({'outdir': args.outdir})
    mkdir_p(args.outdir)
    param.update(PottsParam.process_args(args, param, log))
    log("")
    gpuparam, gpus = GPUParam.process_args(args, param, 2*nloop, log)
    param.update(gpuparam)
    log("")
    param.update(SeqParam.process_args(args, param, gpus, log))
    log("")

    p = param

    if param.sseq_l:
        pass
    elif param.startseq is not None:
        log("Loading small seq buffer with startseq")
        for gpu in gpus:
            gpu.resetSeqs(param.startseq)
    elif param.lseq_l:
        raise Exception("Error: large seq buffer loaded, but need small buffer")
    else:
        raise Exception("Error: To benchmark, must either supply startseq or "
                        "load seqs into small seq buffer")


    log("Benchmark")
    log("=========")
    log("")
    log("Benchmarking MCMC for {} loops".format(nloop))
    import time
    
    #initialize
    for gpu in gpus:
        gpu.setBuf('J main', param.couplings)
    
    #warmup
    log("Warmup run...")
    for gpu in gpus:
        gpu.calcEnergies('small', 'main')
    for i in range(nloop):
        for gpu in gpus:
            gpu.runMCMC()
    for gpu in gpus:
        gpu.wait()
    
    #timed run
    log("Timed run...")
    start = time.clock()
    for i in range(nloop):
        for gpu in gpus:
            gpu.runMCMC()
    for gpu in gpus:
        gpu.wait()
    end = time.clock()

    log("Elapsed time: ", end - start, )
    log("Time per loop: ", (end - start)/nloop)
    steps_per_second = (p.nwalkers*nloop*p.nsteps*p.L)/(end-start)
    log("MC steps per second: {:g}".format(steps_per_second))

def equilibrate():
    if sseq_l is None and lseq_l is None:
        log("Initializing all sequences to start seq")
        for gpu in gpus:
            gpu.resetSeqs(startseq)

    for gpu in gpus:
        gpu.setBuf('J main', couplings)

    mkdir_p(os.path.join(outdir, 'equilibration'))
    mkdir_p(os.path.join(outdir, 'eqseqs'))
    mkdir_p(os.path.join(outdir, 'eqen'))

    log("Equilibrating ...")
    for j in range(nloop/args.trackequil):
        for i in range(args.trackequil):
            for gpu in gpus:
                gpu.runMCMC()
        for gpu in gpus:
            gpu.calcBimarg('small')
            gpu.calcEnergies('small', 'main')
        res = readGPUbufs(['bi main', 'seq small', 'E small'], gpus)
        bimarg_model = meanarr(res[0])
        sampledseqs, sampledenergies = concatenate(res[1]), concatenate(res[2])
        save(os.path.join(outdir, 'eqen', 'en-{}'.format(j)), sampledenergies)
        seqload.writeSeqs(os.path.join(outdir, 'eqseqs', 'seqs-{}'.format(j)), 
                          sampledseqs, alpha)
        save(os.path.join(outdir, 'equilibration', 'bimarg_{}'.format(j)), 
             bimarg_model)

    bicount = sumarr(readGPUbufs(['bicount'], gpus)[0])

    #get summary statistics and output them
    rmsd = sqrt(mean((bimarg_target - bimarg_model)**2))
    ssr = sum((bimarg_target - bimarg_model)**2)
    wdf = sum(bimarg_target*abs(bimarg_target - bimarg_model))
    writeStatus('.', rmsd, ssr, wdf, bicount, bimarg_model, 
                couplings, [sampledseqs], startseq, sampledenergies)

def subseqFreq():
    add('--fixpos', help="dd")

    #load seqs to gpu buffer

    #fix positions
    for gpu in gpus:
        gpu.setBuf('fixpos', fixedpos)

    #equilibrate a little
    for i in range(equiltime):
        for gpu in gpus:
            gpu.runMCMC()
    
    f = zeros(nseq, dtype=float64)

    for i in range(niter):
        #simulate for a little time
        for i in range(sampletime):
            for gpu in gpus:
                gpu.runMCMC()

        seq = getGPUBufs('seq small', seq)
        origEs = getGPUBufs('E small', gpus)
        origseqs = [seq.copy() for s in seq]

        for n in range(nseq):
            for seq, g in zip(seqs, gpus):
                seq[:,fixedpos] = origseq[n,fixedpos]
                g.setBuf('seq small', seq)
                g.getEnergies()
            energies = getGPUBufs('E small', gpus)

            f += exp(energies - origenergies)

        for seq, g in zip(origseqs, gpus):
            g.setBuf('seq small', seq)
    
    # combine repeated subsequences 
    subseq = seqs[:,fixedpos]
    u, inds, counts = unique(subseq.view('S{}'.format(len(fixedpos))), 
                             return_index=True, return_counts=True)
    funique = zeros(len(u))
    for fi,ind in zip(f, inds):
        funique[ind] += fi
    # when combining repeated subsequences, need to downweight repeats
    funique = funique/counts
    
    #normalize and return
    funique = funique/sum(funique)
    return u.view('u1'), funique

################################################################################

class CLInfoAction(argparse.Action):
    def __init__(self, option_strings, dest=argparse.SUPPRESS, 
                 default=argparse.SUPPRESS, help=None):
        super(CLInfoAction, self).__init__(option_strings=option_strings,
            dest=dest, default=default, nargs=0, help=help)
    def __call__(self, parser, namespace, values, option_string=None):
        printGPUs(print)
        parser.exit()

class GPUParam:
    @staticmethod
    def add_arguments(parser):
        group = parser.add_argument_group('GPU options')
        add = group.add_argument
        add('--nwalkers', type=uint32, required=True,
            help="Number of MC walkers")
        add('--nsteps', type=uint32, default=16,
            help="number of MC steps per kernel call, in multiples of L")
        add('--wgsize', type=int, default=256, 
            help="GPU workgroup size")
        add('--gpus', 
            help="GPUs to use (comma-sep list of platforms #s, eg '0,0')")
        add('--profile', action='store_true', 
            help="enable OpenCL profiling")
        # nlargebuf should be specified elsewhere (needed in process_args)
        #add('--nlargebuf', type=uint32, default=1
        #         help='size of large seq buffer, in multiples of nwalkers')

    @staticmethod
    def process_args(args, param, rngPeriod, log):
        log("GPU setup")
        log("---------")

        p = attrdict({'nwalkers': args.nwalkers,
                      'nlargebuf': args.nlargebuf,
                      'nsteps': args.nsteps,
                      'wgsize': args.wgsize,
                      'gpuspec': args.gpus,
                      'profile': args.profile})
        allparam = attrdict(p.copy())
        allparam.update(param)
        
        scriptfile = os.path.join(scriptPath, "metropolis.cl")
        gpus = initGPUs(scriptPath, scriptfile, allparam, rngPeriod, log)

        log("Work Group Size: {}".format(p.wgsize))
        log(("Can run {} MCMC walkers in parallel over {} GPUs, with {} MC "
            "steps per kernel call ({}*{})").format(p.nwalkers, len(gpus), 
            param.L*p.nsteps, param.L, p.nsteps))
        if p.profile:
            log("Profiling Enabled")

        return p, gpus

class NewtonParam:
    @staticmethod
    def add_arguments(parser):
        group = parser.add_argument_group('Newton Step options')
        add = group.add_argument
        add('--bimarg', required=True,
            help="Target bivariate marginals (npy file)")
        add('--mcsteps', type=uint32, required=True, 
            help="Number of rounds of MCMC generation") 
        add('--newtonsteps', default='128', type=uint32, 
            help="Number of newton steps per round.")
        add('--gamma', type=float32, required=True, 
            help="Initial step size")
        add('--damping', default=0.001, type=float32, 
            help="Damping parameter")
        add('--jclamp', default=0, type=float32, 
            help="Clamps maximum change in couplings per newton step")
        add('--preopt', action='store_true', 
            help="Perform a round of newton steps before first MCMC run") 

    @staticmethod
    def process_args(args, param, log):
        log("Newton Solver Setup")
        log("-------------------")
        mcmcsteps = args.mcsteps  
        log("Running {} Newton-MCMC rounds".format(mcmcsteps))

        param = {'mcmcsteps': args.mcsteps,
                 'newtonSteps': args.newtonsteps,
                 'gamma0': args.gamma,
                 'pcdamping': args.damping,
                 'jclamp': args.jclamp,
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

        p['bimarg'] = bimarg
        return p

class PottsParam:
    @staticmethod
    def add_arguments(parser):
        group = parser.add_argument_group('Potts Model options')
        add = group.add_argument
        add('--alpha', required=True,
            help="Alphabet, a sequence of letters")
        add('--couplings', 
            help="One of 'zero', 'logscore', or a filename")
        add('--L', help="sequence length") 
        # also depends on 'seqmodel' arg, but must be supplied by other means

    # this will load the sequences direct to the gpu
    @staticmethod
    def process_args(args, param, log):
        log("Potts Model Setup")
        log("-----------------")
        bimarg = param.bimarg

        # we try to infer L and nB from any values given. The possible sources
        # * command line options -L and -nB
        # * from bivariate_target dimensions
        # * from coupling dimensions
        
        alpha = args.alpha
        L, nB = args.L, len(alpha)
        L, nB = PottsParam.updateLnB(L, nB, param.L, param.nB, 'bimarg')
        
        # next try to get couplings (may determine L, nB)
        couplings, L, nB = PottsParam.getCouplings(args, L, nB, bimarg, log)
        # we should have L and nB by this point

        log("alphabet: {}".format(alpha))
        log("nBases {}  seqLen {}".format(nB, L))
        log("Couplings: " + printsome(couplings) + "...")

        return attrdict({'L': L, 'nB': nB, 'alpha': alpha, 
                         'couplings': couplings})

    @staticmethod
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

    @staticmethod
    def getCouplings(args, L, nB, bimarg, log):
        couplings = None

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
        L, nB = PottsParam.updateLnB(L, nB, L2, nB2, 'couplings')

        if couplings is None:
            raise Exception("Could not find couplings. Use either the "
                            "'couplings' or 'seqmodel' options.")
        #switch to 'even' fieldless gauge for nicer output
        h0, J0 = zeroGauge(zeros((L,nB)), couplings)
        couplings = fieldlessGaugeEven(h0, J0)[1]

        return couplings, L, nB

# need to separate out the loading of the sequences from file and sending to GPU.
# it might be costly in terms of memory, but sometimes we need to know the sequence set size before initializing the gpu
#maybe can get around this by returning the file handle instead
class SeqParam:
    @staticmethod
    def add_arguments(parser):
        group = parser.add_argument_group('Sequence options')
        add = group.add_argument
        add('--startseq', help="Starting sequence. May be 'rand'") 
        add('--seqs', help="File containing sequences to pre-load to GPU") 

    # this will load the sequences direct to the gpu
    @staticmethod
    def process_args(args, param, gpus, log):
        log("Sequence Setup")
        log("--------------")

        L, nB = param.L, param.nB
        alpha = param.alpha

        # check if we were asked to generate sequences
        sseq_l, lseq_l = None, None
        startseq = None
        if args.seqmodel in ['rand', 'logscore']:
            if args.seqs is not None:
                raise Exception("Cannot specify both seqs and "
                                "seqmodel=[rand, logscore]")
            startseq = SeqParam.generateSequences(args.seqmodel, gpus, 
                                                  L, nB, log)
            startseq_origin = 'generated ' + args.seqmodel
            seqmodeldir = None
            sseq_l, lseq_l = True, True
        else:
            seqmodeldir = args.seqmodel

        # try to load sequence files
        loadSequenceFile = SeqParam.loadSequenceFile
        if args.seqs in ['zero', 'logscore']:
            SeqParam.generateSequences(args.seqs, gpus, L, nB, log)
            sseq_l, lseq_l = True, True
        elif args.seqs:
            sseq_l, lseq_l = loadSequenceFile(args.seqs, alpha, gpus, log)
        elif seqmodeldir is not None:
            fn = os.path.join(seqmodeldir, 'seq')
            fns = [os.path.join(seqmodeldir, 'seq-{}'.format(n)) 
                   for n in range(len(gpus))]
            if os.path.isfile(fn):
                sseq_l, lseq_l = loadSequenceFile(fn, alpha, gpus, log)
            elif all([os.path.isfile(fni) for fni in fns]):
                sseq_l, lseq_l = SeqParam.loadSequenceDir(
                                             seqmodeldir, alpha, gpus)

        # try to get start seq
        if args.startseq:
            if args.startseq == 'rand':
                startseq = randint(0, nB, size=L).astype('<u1')
                startseq_origin = 'random'
            else:
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

        if not sseq_l and not lseq_l:
            log("No sequences loaded into GPU")

        if startseq is not None:
            log("Start seq ({}): {}".format(startseq_origin, 
                                           "".join(alpha[x] for x in startseq)))
        else:
            log("No start seq supplied")

        return attrdict({'startseq': startseq, 
                         'sseq_l': sseq_l, 
                         'lseq_l': lseq_l})

    @staticmethod
    def generateSequences(gentype, gpus, L, nB, log):
        # fill all gpu buffers with generated sequences, and return one
        # of them as the start sequence

        if gentype == 'zero': 
            log("Generating random sequences...")
            for gpu in gpus:
                nseq = gpu.nseq['large']
                seqs = numpy.random.randint(0,nB,size=(nseq, L)).astype('<u1')
                gpu.setBuf('seq large', seqs)
                gpu.setBuf('seq small', seqs[:gpu.nseq['small']])
            startseq = seqs[-1]
        elif gentype == 'logscore': 
            log("Generating logscore-independent sequences...")
            cumprob = cumsum(marg, axis=1)
            cumprob = cumprob/(cumprob[:,-1][:,newaxis]) #correct fp errors?
            for gpu in gpus:
                seqs = array([searchsorted(cp, rand(pnseq)) 
                              for cp in cumprob], dtype='<u1').T.astype('<u1')
                gpu.setBuf('seq large', seqs)
                gpu.setBuf('seq small', seqs[:nwalkers])
            startseq = seqs[-1]
        return startseq

    @staticmethod
    def loadSequenceDir(sdir, alpha, gpus):
        # tries to load sequences. It will choose which GPU
        # buffer to load them to depending on the file sizes.
        # 2 possibilities: seqs for small buf, seqs for large buf
        #returns whether it loaded the small and large buffers

        gpu_smallbuf_seqs = gpus[0].nseq['small']
        gpu_largebuf_seqs = gpus[0].nseq['large']

        log("Loading sequences from dir {}".format(sdir))

        bufname, bufsize = None, None
        for n,gpu in enumerate(gpus):
            seqfile = os.path.join(args.prestart, 'seqs-{}'.format(n))
            seqs = seqload.loadSeqs(sdir, names=alpha)[0].astype('<u1')
            
            if bufsize is None:
                if seqs.shape[0] == gpu_largebuf_seqs:
                    bufname, bufsize = 'large', gpu_largebuf_seqs
                    log("Loading gpu's large seq buffer")
                elif seqs.shape[0] == gpu_smallbuf_seqs:
                    bufname, bufsize = 'small', gpu_smallbuf_seqs
                    log("Loading gpu's small seq buffer")
                else:
                    raise Exception(("Expected either {} or {} sequences,"
                                     "got {}").format(gpu_largebuf_seqs, 
                                      gpu_smallbuf_seqs, seqs.shape[0]))

            if seqs.shape[0] != bufsize:
                raise Exception("Expected {} sequences, got {}".format(
                                bufsize, seqs.shape[0]))

            gpu.setBuf('seq ' + bufname, seqs)

        return bufname == 'small', bufname == 'large'

    @staticmethod
    def loadSequenceFile(sfile, alpha, gpus, log):
        log("Loading sequences from file {}".format(sfile))
        total_smallbuf_seqs = sum(g.nseq['small'] for g in gpus)
        total_largebuf_seqs = sum(g.nseq['large'] for g in gpus)

        seqs = seqload.loadSeqs(sfile, names=alpha)[0].astype('<u1')

        if seqs.shape[0] == total_largebuf_seqs:
            bufname = 'large'
            log("Loading gpu's large seq buffer")
        elif seqs.shape[0] == total_smallbuf_seqs:
            bufname = 'small'
            log("Loading gpu's small seq buffer")
        else:
            raise Exception(("Expected either {} or {} sequences,"
                             "got {}").format(total_largebuf_seqs, 
                              total_smallbuf_seqs, seqs.shape[0]))

        for s, gpu in zip(seqs.split(len(gpus)), gpus):
            gpu.setBuf('seq ' + bufname, s)

        return bufname == 'small', bufname == 'large'

class SampleParam:
    @staticmethod
    def add_arguments(parser):
        group = parser.add_argument_group('Sampling options')
        add = group.add_argument
        add('--equiltime', type=uint32, required=True, 
            help="Number of MC kernel calls to equilibrate")
        add('--sampletime', type=uint32, required=True, 
            help="Number of MC kernel calls between samples")
        add('--nsamples', type=uint32, required=True, 
            help="Number of sequence samples")
        add('--trackequil', type=uint32, default=0,
            help='Save bimarg every TRACKEQUIL steps during equilibration')

    @staticmethod
    def process_args(args, param, log):
        p = attrdict({'nloop': args.equiltime,
                      'nsampleloops': args.sampletime,
                      'nsamples': args.nsamples,
                      'trackequil': args.trackequil})

        if p.nsamples == 0:
            raise Exception("nsamples must be at least 1")

        log("MCMC Sampling Setup")
        log("-------------------")
        log(("In each MCMC round, running {} GPU kernel calls then sampling "
             "every {} kernel calls to get {} samples").format(p.nloop, 
                                                 p.nsampleloops, p.nsamples))

        if p.trackequil != 0:
            if p.nloop%p.trackequil != 0:
                raise Exception("Error: trackequil must be a divisor of nloop")
            log("Tracking equilibration every {} loops.".format(p.trackequil))

        return p

def main(args):
    actions = {
      'inverseIsing':     inverseIsing,
      'getEnergies':    getEnergies,
      'benchmark':      MCMCbenchmark,
      #'measureFPerror': measureFPerror
     }

    
    descr = 'Perform biophysical Potts Model calculations on the GPU'
    parser = argparse.ArgumentParser(description=descr, add_help=False)
    add = parser.add_argument
    add('action', choices=actions.keys(), nargs='?', default=None,
        help="Computation to run")
    add('--config', help="Config file with command line args (INI format)")
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
    
    actions[known_args.action](remaining_args, print)

main(sys.argv[1:])

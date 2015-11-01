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
from mcmcGPU import initGPUs

################################################################################
# Set up enviroment and some helper functions

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
    marg = array([sum(ff[0],axis=1)] + [sum(ff[n],axis=0) for n in range(L-1)])
    return marg/(sum(marg,axis=1)[:,newaxis]) # correct any fp errors

################################################################################

def solvePotts(parser, args, log):
    GPUParam.add_arguments(parser)
    SeqParam.add_arguments(parser)
    NewtonParam.add_arguments(parser)
    SampleParam.add_arguments(parser)
    PottsParam.add_arguments(parser)
    parser.add_argument('--outdir', default='output')
    
    args = parser.parse_args(args)
    args.nlargebuf = args.nsamples

    log("Initialization")
    log("===============")
    log("")
    
    param = attrdict({'outdir': args.outdir})
    mkdir_p(args.outdir)

    param.update(NewtonParam.process_args(args, param, log))
    log("")
    if param.bimarg is not None:
        param['L'], param['nB'] = seqsize_from_param_shape(param.bimarg.shape)
    param.update(PottsParam.process_args(args, param, log))
    log("")
    param, gpus = GPUParam.process_args(args, param, log)
    log("")
    param.update(SampleParam.process_args(args, param, log))
    log("")
    param.update(SeqParam.process_args(args, param, gpus, log))
    log("")

    p = param
    if p.startseq == None and p.preopt == None:
        raise Exception("Starting seq not found")
    # also warn if we loaded seqs, but no preopt done

    log("Action Plan")
    log("===========")
    log("")
    log("Performing Inverse Potts Inference using quasi-Newton MCMC")
    log("")
    log("Running {} Newton-MCMC rounds, with {} parameter update steps per "
        "round.".format(p.mcmcsteps, p.newtonSteps))
    log("")
    log(("In each round, running {} MC walkers for {} equilibration loops then "
         "sampling every {} loops to get {} samples ({} total seqs) with {} MC "
         "steps per loop (Each walker equilibrated a total of {} MC steps)."
         ).format(p.nwalkers, p.nloop, p.nsampleloops, p.nsamples, 
                  p.nsamples*p.nwalkers, p.nsteps*L, p.nsteps*L*p.nloop))
    log("")
    log("MCMC Run")
    log("========")
    doFit(startseq, couplings, gpus)

def getEnergies(parser, args, log):
    GPUParam.add_arguments(parser)
    SeqParam.add_arguments(parser)

    args = parser.parse_args(args)

    return

    for gpu in gpus:
        gpu.calcEnergies('small', 'main')

    es = readGPUbufs(['E small'], gpus)

def MCMCbenchmark(args):

    gpuPrm, seqPrm, benchmarkPrm

    log("Benchmark")
    log("=========")
    log("")
    log("Benchmarking MCMC for {} loops".format(nloop))
    import time
    
    #initialize
    for gpu in gpus:
        gpu.resetSeqs(startseq)
        gpu.setBuf('J main', couplings)
    
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
    steps_per_second = (nwalkers*nloop*nsteps*L)/(end-start)
    log("MC steps per second: {:g}".format(steps_per_second))

def equilibrate():
    if sseq_loaded is None and lseq_loaded is None:
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

################################################################################

class CLInfoAction(argparse.Action):
    def __init__(self, option_strings, dest=argparse.SUPPRESS, 
                 default=argparse.SUPPRESS, help=None):
        super(CLInfoAction, self).__init__(option_strings=option_strings,
            dest=dest, default=default, nargs=0, help=help)
    def __call__(self, parser, namespace, values, option_string=None):
        printGPUs()
        parser.exit()

class GPUParam:
    @staticmethod
    def add_arguments(parser):
        group = parser.add_argument_group('GPU options')
        add = group.add_argument
        add('--nwalkers', type=uint32, required=True,
            help="Number of MC walkers")
        # nlargebuf should be specified elsewhere (needed in process_args)
        #add('--nlargebuf', type=uint32, default=1
        #         help='size of large seq buffer, in multiples of nwalkers')
        add('--nsteps', type=uint32, default=1,
                 help="number of MC steps per kernel call, in multiples of L")
        add('--wgsize', type=int, default=256, help="GPU workgroup size")
        add('--gpus', help=("GPUs to use (comma-sep list of platforms #s, "
                           "eg '0,0')"))
        add('--profile', action='store_true', help="enable OpenCL profiling")

    @staticmethod
    def process_args(args, param, log):
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
        gpus = initGPUs(scriptPath, scriptfile, allparam, log)
        return p, gpus

class NewtonParam:
    @staticmethod
    def add_arguments(parser):
        group = parser.add_argument_group('Newton Step options')
        add = group.add_argument
        add('--bimarg', required=True, help="Target bivariate marginals (npy file)")
        add('--mcsteps', type=uint32, required=True, help="Number of rounds of MCMC generation") 
        add('--newtonsteps', default='128', help="Number of newton steps per round.")
        add('--gamma', type=float32, required=True, help="Initial step size")
        add('--damping', default=0.001, help="Damping parameter")
        add('--Jclamp', default=None, help="Clamps maximum change in couplings per newton step")
        add('--preopt', action='store_true', help="Perform a round of newton steps before first MCMC run") 

    @staticmethod
    def process_args(args, param, log):
        log("Newton Update Setup")
        log("-------------------")
        mcmcsteps = args.mcsteps  
        log("Running {} Newton-MCMC rounds".format(mcmcsteps))

        param = {'mcmcsteps': args.mcsteps,
                 'newtonSteps': args.newtonsteps,
                 'gamma0': args.gamma,
                 'pcDamping': args.damping,
                 'Jclamp': args.Jclamp,
                 'preopt': args.preopt }
        p = attrdict(param)

        cutoffstr = ('dJ clamp {}'.format(p.Jclamp) if p.Jclamp != None 
                     else 'no dJ clamp')
        log(("Updating J locally with gamma = {}, {}, and pc-damping {}. "
             "Running {} Newton update steps per round.").format(
              p.gamma0, cutoffstr, p.pcDamping, p.newtonSteps))

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
        add('--alpha', help="Alphabet, a sequence of letters", required=True)
        add('--couplings', default='none', required=True,
                  help="One of 'zero', 'logscore', or a filename")
        add('--L', help="sequence length") 

    # this will load the sequences direct to the gpu
    @staticmethod
    def process_args(args, param, log):
        log("Potts Model Setup")
        log("--------------------")

        # we try to infer L and nB from any values given. The possible sources
        # * command line options -L and -nB
        # * from bivariate_target dimensions
        # * from coupling dimensions
        
        alpha = args.alpha
        L, nB = args.L, len(alpha)
        L, nB = PottsParam.updateLnB(L, nB, param.L, param.nB, 'bimarg')
        
        # next try to get couplings (may determine L, nB)
        couplings, L, nB = PottsParam.getCouplings(args, L, nB, param.bimarg, log)
        # we should have L and nB by this point


        log("alphabet: {}".format(alpha))
        log("nBases {}  seqLen {}".format(nB, L))
        log("Couplings: " + printsome(couplings) + "...")

        return attrdict({'L': L, 'nB': nB, 'alpha': alpha, 'couplings': couplings})

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
                    raise Exception("Need bivariate marginals to generate logscore couplings")
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
            if isfile(fn):
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

class SeqParam:
    @staticmethod
    def add_arguments(parser):
        group = parser.add_argument_group('Sequence options')
        add = group.add_argument
        add('--seq', help="Primary (starting) sequence. May be 'rand'") 
        add('--seqs', help="File containing sequences to pre-load to GPU") 
        add('--seqmodel', default='none',
          help=("One of 'zero', 'logscore', or a directory name. Generates "
                "or loads 'couplings', 'seq' and 'seqs'.") )

    # this will load the sequences direct to the gpu
    @staticmethod
    def process_args(args, param, gpus, seqsize,  log):
        L, nB = args.L, len(alpha)

        # check if we were asked to generate sequences
        sseq_loaded, lseq_loaded = None, None
        startseq = None
        if args.seqmodel in ['rand', 'logscore']:
            if args.seqs is not None:
                raise Exception("Cannot specify both seqs and "
                                "seqmodel=[rand, logscore]")
            startseq = SeqParam.generateSequences(args.seqmodel, L, nB)
            startseq_origin = 'generated ' + args.seqmodel
            seqmodeldir = None
        else:
            seqmodeldir = args.seqmodel

        # try to load sequence files
        if args.seqs:
            sseq_loaded, lseq_loaded = SeqParam.loadSequenceFile(args.seqs)
        elif seqmodeldir is not None:
            fn = os.path.join(seqmodeldir, 'seq')
            fns = [os.path.join(seqmodeldir, 'seq-{}'.format(n)) 
                   for n in range(len(gpus))]
            if isfile(fn):
                sseq_loaded, lseq_loaded = SeqParam.loadSequenceFile(
                                                        fn, alpha, gpus)
            elif all([isfile(fni) for fni in fns]):
                sseq_loaded, lseq_loaded = SeqParam.loadSequenceDir(
                                             seqmodeldir, alpha, gpus)

        # try to get start seq
        if args.seq:
            if args.seq == 'rand':
                startseq = randint(0, nB, size=L).astype('<u1')
                startseq_origin = 'random'
            else:
                startseq = array(map(alpha.index, args.seq), dtype='<u1')
                startseq_origin = 'supplied'
        elif seqmodeldir is not None:
            fn = os.path.join(seqmodeldir, 'startseq')
            if isfile(fn):
                log("Reading startseq from file {}".format(fn))
                with open(fn) as f:
                    startseq = f.readline().strip()
                    startseq = array(map(alpha.index, startseq), dtype='<u1')
                startseq_origin = 'from file'

        if startseq is not None:
            log("Start seq ({}): {}".format(startseq_origin, "".join(alpha[x] 
                                                          for x in startseq)))
        else:
            log("No start seq supplied")

        return (L, nB, alpha, couplings, startseq, sseq_loaded, lseq_loaded)

    @staticmethod
    def generateSequences(gentype, gpus):
        # fill all gpu buffers with generated sequences, and return one
        # of them as the start sequence

        if gentype == 'zero': 
            log("Generating random sequences...")
            for gpu in gpus:
                seqs = numpy.random.randint(0,nB,size=(pnseq, L)).astype('<u1')
                gpu.setBuf('seq large', seqs)
                gpu.setBuf('seq small', seqs[:nwalkers])
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

        gpu_largebuf_seqs = nsamples*nwalkers/len(gpus)
        gpu_smallbuf_seqs = nwalkers/len(gpus)

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
    def loadSequenceFile(sfile, alpha, gpus):
        log("Loading sequences from file {}".format(sfile))
        total_smallbuf_seqs = nwalkers
        total_largebuf_seqs = nsamples*nwalkers

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
        add('--equiltime', type=uint32, required=True, help="Number of MC kernel calls to equilibrate")
        add('--sampletime', type=uint32, required=True, help="Number of MC kernel calls between samples")
        add('--nsamples', type=uint32, required=True, help="Number of sequence samples")
        add('--trackequil', type=uint32, default=0,
                 help='During equilibration, save bimarg every TRACKEQUIL kernel calls')

    @staticmethod
    def process_args(args, log):
        nloop = args.nloop  
        nsampleloops = args.nsampleloops    
        nsamples = args.nsamples  
        trackequil = args.trackequil

        if nsamples == 0:
            raise Exception("nsamples must be at least 1")

        log("Sequence Sampling")
        log("-----------------")
        log(("Running {} GPU loops then sampling every {} loops "
             "to get {} samples").format( nloop, nsampleloops, nsamples))

        if trackequil != 0:
            if nloop%trackequil != 0:
                raise Exception("Error: trackequil must be a divisor of nloop")
            log("Tracking equilibration every {} loops.".format(trackequil))

        return (nloop, nsamples, nsampleloops, trackequil)

def main(args):
    actions = {
      'solvePotts':     solvePotts,
      'getEnergies':    getEnergies,
      'benchmark':      MCMCbenchmark,
      #'measureFPerror': measureFPerror
     }

    parser = argparse.ArgumentParser(description='IvoGPU', add_help=False)
    parser.add_argument('action', choices=actions.keys(), help="Computation to run")
    parser.add_argument('--config')
    parser.add_argument('--clinfo', action=CLInfoAction)

    known_args, remaining_args = parser.parse_known_args(args)
    
    parser = argparse.ArgumentParser(description='IvoGPU')
    actions[known_args.action](parser, remaining_args, print)

main(sys.argv[1:])

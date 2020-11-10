#!/usr/bin/env python3 -u
#
# Copyright 2020 Allan Haldane.
#
# This file is part of Mi3-GPU.
#
# Mi3-GPU is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3 of the License.
#
# Mi3-GPU is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Mi3-GPU.  If not, see <http://www.gnu.org/licenses/>.
#
#Contact: allan.haldane _AT_ gmail.com

import sys, os, errno, time, datetime, socket, signal, atexit, glob, argparse
from pathlib import Path
import numpy as np
from numpy.random import randint, rand
from scipy.special import logsumexp
import pyopencl as cl
import pyopencl.array as cl_array
import json
import pkg_resources  # part of setuptools
version = pkg_resources.require("mi3gpu")[0].version

import mi3gpu
import mi3gpu.NewtonSteps
from mi3gpu.utils.seqload import loadSeqs, writeSeqs
from mi3gpu.utils.changeGauge import fieldlessGaugeEven
from mi3gpu.utils.potts_common import printsome, getLq, getUnimarg
from mi3gpu.mcmcGPU import (setup_GPU_context, initGPU, wgsize_heuristic, 
                            printGPUs)
from mi3gpu.node_manager import GPU_node

try:
    from shlex import quote as cmd_quote
except ImportError:
    from pipes import quote as cmd_quote

MPI = None
def setup_MPI():
    global MPI, mpi_comm, mpi_rank
    global MPI_multinode_controller, MPI_GPU_node, MPI_worker

    from mpi4py import MPI

    mpi_comm = MPI.COMM_WORLD
    if mpi_comm.Get_size() == 1:
        MPI = None
    else:
        mpi_rank = mpi_comm.Get_rank()
        from mpi_manager import (MPI_multinode_controller,
                                 MPI_GPU_node, MPI_worker)

################################################################################
# Set up enviroment and some helper functions

progname = 'Mi3.py'

scriptPath = Path(mi3gpu.__file__).parent
scriptfile = scriptPath / "mcmc.cl"

class attrdict(dict):
    def __getattr__(self, attr):
        if attr.startswith('_'):
            return super().__getattr__(attr)
        try:
            return dict.__getitem__(self, attr)
        except KeyError:
            return None

#identical calculation as CL kernel, but with high precision (to check fp error)
def getEnergiesMultiPrec(s, couplings):
    from mpmath import mpf, mp
    mp.dps = 32
    couplings = [[mpf(float(x)) for x in r] for r in couplings]
    pairenergy = [mpf(0) for n in range(s.shape[0])]
    s = s.astype('i4')
    for n,(i,j) in enumerate([(i,j) for i in range(L-1) for j in range(i+1,L)]):
        r = couplings[n]
        cpl = (r[b] for b in (q*s[:,i] + s[:,j]))
        pairenergy = [x+n for x,n in zip(pairenergy, cpl)]
    return pairenergy

################################################################################

def optionRegistry():
    options = []
    add = lambda opt, **kwds: options.append((opt, kwds))

    # option used by both potts and sequence loaders, designed
    # to load in the output of a previous run
    add('init_model', default='independent',
        help=("One of 'zero', 'independent', or a directory name. Generates or "
              "loads 'alpha', 'couplings', 'seedseq' and 'seqs', if not "
              "otherwise supplied.") )
    add('outdir', type=Path, default='output', help='Output Directory')
    add('finish', type=Path, help='Dir. of an unfinished run to finish')
    add('config', #is_config_file_arg=True,
                  help='config file to load arguments from')
    add('rngseed', type=np.uint32, help='random seed')
    add('log', action='store_true', help='log program output')

    # GPU options
    add('nwalkers', type=np.uint32,
        help="Number of MC walkers")
    add('nsteps', type=np.uint32, default=2048,
        help="number of mc steps per kernel call")
    add('wgsize', default=512, help="GPU workgroup size")
    add('gpus',
        help="GPUs to use (comma-sep list of platforms #s, eg '0,0')")
    add('profile', action='store_true',
        help="enable OpenCL profiling")
    add('nlargebuf', type=np.uint32, default=1,
        help='size of large seq buffer, in multiples of nwalkers')
    add('measurefperror', action='store_true',
        help="enable fp error calculation")

    # Newton options
    add('bimarg', required=True,
        help="Target bivariate marginals (npy file)")
    add('mcsteps', type=np.uint32, default=64,
        help="Number of rounds of MCMC generation")
    add('newtonsteps', default=1024, type=np.uint32,
        help="Initial number of newton steps per round.")
    add('newton_delta', default=32, type=np.uint32,
        help="Newton step number tuning scale")
    add('fracNeff', type=np.float32, default=0.9,
        help="stop coupling updates after Neff/N = fracNeff")
    add('gamma', type=np.float32, default=0.0004,
        help="Initial step size")
    add('damping', default=0.001, type=np.float32,
        help="Damping parameter")
    add('reg', default=None,
        help="regularization format")
    add('preopt', action='store_true',
        help="Perform a round of newton steps before first MCMC run")
    add('reseed',
        choices=['none', 'single_best', 'single_random', 'single_indep',
                 'independent', 'uniform', 'msa'],
        default='single_indep',
        help="Strategy to reset walkers after each MCMC round")
    add('seedmsa', default=None,
        help="seed used of reseed=msa")
    add('distribute_jstep', choices=['head_gpu', 'head_node', 'all'],
        default='all',
        help="how to split newton step computation across GPUs")

    # Potts options
    add('alpha', required=True,
        help="Alphabet, a sequence of letters")
    add('couplings',
        help="One of 'zero', 'independent', or a filename")
    add('L', help="sequence length", type=int)

    # Sequence options
    add('seedseq', help="Starting sequence. May be 'rand'")
    add('seqs', help="File containing sequences to pre-load to GPU")
    add('seqs_large', help="File containing sequences to pre-load to GPU")
    add('indep_marg',
        help="marg (uni or bi to convert to uni) used to generate "
             "site-independent sequences")

    # Sampling Param
    add('equiltime', default='auto',
        help="Number of MC kernel calls to equilibrate")
    add('min_equil', default=64, type=int,
        help="minimum MC calls to equilibrate when using 'equiltime=auto'")
    add('trackequil', type=np.uint32, default=0,
        help='Save "--tracked" data every TRACKEQUIL steps during mcmc')
    add('tracked', default='bim,E',
        help='Data saved to "equilibration" dir when trackequil is enabled. '
             'Comma separated allowed options: "bim", "E", "seq"')
    add('tempering',
        help='optional inverse Temperature schedule')
    add('nswaps_temp', type=np.uint32, default=128,
        help='optional number of pt swaps')

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

def setup_seed(args, p, log):
    if args.rngseed is not None:
        seed = args.rngseed
    else:
        seed = np.frombuffer(os.urandom(4), dtype='u4')[0]

    # set up numpy rng seed on head node (used in seq gen)
    np.random.seed(seed)
    # set rngseed param, used in mcmcGPU for randomized mutation pos
    p['rngseed'] = seed + 1  # +1 just so rng for seq gen is diff from mcmc
    log("Using random seed {}".format(p.rngseed))
    log("")

def describe_tempering(args, p, log):
    if p.tempering is not None:
        if len(p.tempering) == p.nwalkers:
            msg = ("The walkers are assigned temperatures from file {}"
                  ).format(args.tempering)
        else:
            msg = ("The walkers are divided into {} temperature groups ({})"
                  ).format(len(p.tempering), args.tempering)

        log(("Parallel tempering: {}, and neighbor temperatures are "
             "swapped {} times after every MCMC loop. The low-temperature "
             "B is {}").format(msg, p.nswaps, np.max(p.tempering)))

def print_node_startup(log, orig_args):
    log("Hostname:   {}".format(socket.gethostname()))
    log("Start Time: {}".format(datetime.datetime.now()))
    if 'PBS_JOBID' in os.environ:
        log("Job name:   {}".format(os.environ['PBS_JOBID']))

    log("")
    log("Command line arguments:")
    log(" ".join(cmd_quote(a) for a in orig_args))
    log("")
    log("Mi3-GPU Version:", version)
    log("")

def setup_exit_hook(log):
    def exiter():
        if MPI:
            mpi_comm.Abort()
        log("Exited at {}".format(datetime.datetime.now()))

    atexit.register(exiter)

    # Normal exit when killed
    signal.signal(signal.SIGTERM, lambda signum, stack_frame: exit(1))

################################################################################

def divideWalkers(nwalkers, ngpus, log, wgsize=None):
    n_max = (nwalkers-1)//ngpus + 1
    nwalkers_gpu = [n_max]*(ngpus-1) + [nwalkers - (ngpus-1)*n_max]
    if wgsize is not None and nwalkers % (ngpus*wgsize) != 0:
        log("Warning: number of MCMC walkers is not a multiple of "
            "wgsize*ngpus, so there are idle work units.")
    return nwalkers_gpu

def worker_GPU_main():
    def log(*x):
        pass

    p = mpi_comm.bcast(None, root=0)

    # see setup_GPUs_MPI for head node setup
    clinfo, gpudevs, ptx = setup_GPU_context(scriptPath, scriptfile, p, log)

    mpi_comm.gather(len(gpudevs), root=0)

    gpu_ids = mpi_comm.scatter(None, root=0)

    gpus = [initGPU(id, clinfo, dev, nwalk, p, log)
            for dev,(id, nwalk) in zip(gpudevs, gpu_ids)]

    worker = MPI_worker(mpi_rank, gpus)
    worker.listen()

def setup_GPUs_MPI(p, log):
    # setup context for head node
    clinfo, gpudevs, cllog = setup_GPU_context(scriptPath, scriptfile, p, log)
    with open(p.outdir / 'ptx', 'wt') as f:
        f.write(cllog[1])
        f.write(cllog[0])

    # gather GPU setup info from all nodes
    mpi_comm.bcast(p, root=0)

    node_ngpus = mpi_comm.gather(len(gpudevs), root=0)

    ngpus = sum(node_ngpus)
    gpuwalkers = divideWalkers(p.nwalkers, ngpus, log, p.wgsize)
    gpu_param = list(enumerate(gpuwalkers))
    gpu_param = [[gpu_param.pop(0) for i in range(n)] for n in node_ngpus]

    gpu_param = mpi_comm.scatter(gpu_param, root=0)

    log("Found {} GPUs over {} nodes".format(ngpus, len(node_ngpus)))
    log("Starting GPUs...")
    # initialize head node gpus
    headgpus = [initGPU(id, clinfo, dev, nwalk, p, log)
                for dev,(id, nwalk) in zip(gpudevs, gpu_param)]

    workers = ([GPU_node(headgpus)] +
               [MPI_GPU_node(r+1, n) for r, n in enumerate(node_ngpus[1:])])
    gpus = MPI_multinode_controller(workers)
    log('Running on GPUs:\n' +
        "\n".join('    {}   ({} walkers)'.format(n, nwalk)
                  for n, nwalk in zip(gpus.gpu_list, gpuwalkers)))
    return gpus

def setup_GPUs(p, log):
    if MPI:
        return setup_GPUs_MPI(p, log)

    clinfo, gpudevs, cllog = setup_GPU_context(scriptPath, scriptfile, p, log)
    with open(p.outdir / 'ptx', 'wt') as f:
        f.write(cllog[1])
        f.write(cllog[0])

    ngpus = len(gpudevs)

    gpuwalkers = divideWalkers(p.nwalkers, ngpus, log, p.wgsize)
    gpu_param = enumerate(gpuwalkers)

    log("Found {} GPUs".format(ngpus))
    log("GPU Initialization:")
    headgpus = [initGPU(id, clinfo, dev, nwalk, p, log)
                for dev,(id, nwalk) in zip(gpudevs, gpu_param)]

    gpus = GPU_node(headgpus)
    log('Running on GPUs:\n' +
        "\n".join('    {}   ({} walkers)'.format(n, nwalk)
                  for n, nwalk in zip(gpus.gpu_list, gpuwalkers)))
    return gpus

################################################################################

def inverseIsing(orig_args, args, log):
    descr = ('Inverse Ising inference using a quasi-Newton MCMC algorithm '
             'on the GPU')
    parser = argparse.ArgumentParser(prog=progname + ' inverseIsing',
                                     description=descr)
    addopt(parser, 'GPU options',         'nwalkers nsteps wgsize '
                                          'gpus profile')
    addopt(parser, 'Sequence Options',    'seedseq seqs seqs_large')
    addopt(parser, 'Newton Step Options', 'bimarg mcsteps newtonsteps '
                                          'newton_delta fracNeff '
                                          'damping reg distribute_jstep gamma '
                                          'preopt reseed seedmsa')
    addopt(parser, 'Sampling Options',    'equiltime min_equil '
                                          'trackequil tracked '
                                          'tempering nswaps_temp ')
    addopt(parser, 'Potts Model Options', 'alpha couplings L')
    addopt(parser,  None,                 'init_model outdir log rngseed '
                                          'config finish')

    args = parser.parse_args(args)
    args.measurefperror = False

    # set up output directory and log file
    if args.finish:
        if not args.outdir.is_dir():
            raise ValueError("{} is not a directory".format(args.outdir))

        if args.log:
            n = 1
            while true:
                p = args.outdir / 'log{}'.format(n)
                if not p.exists():
                    break
            logfile = open(p, 'wt')
            log = lambda *s, **kwds: print(*s, file=logfile, flush=True, **kwds)
    else:
        args.outdir.mkdir(parents=True)
        if args.log:
            logfile = open(args.outdir / 'log', 'wt')
            log = lambda *s, **kwds: print(*s, file=logfile, flush=True, **kwds)

    print_node_startup(log, orig_args)

    if MPI:
        log("MPI detected using {} processes".format(mpi_comm.Get_size()))
        log("")

    log("Initialization")
    log("===============")

    if args.finish:
        # search for last completed run (contains a perturbedJ file)
        rundirs = args.finish.glob('run_*')
        rundirs.sort()
        rundir = None
        for fn in reversed(rundirs):
            if (fn / 'perturbedJ.npy').is_file():
                rundir = fn
                break
        if rundir is None:
            raise Exception("Did not find any runs in {}".format(args.finish))

        log("Continuing from {}".format(rundir))
        args.init_model = rundir
        log("")
        with open(args.finish / 'config.json', 'r') as f:
            args = argparse.NameSpace(**json.load(f))
        log("loaded parameters:")
        log(vars(args))

        args.outdir = args.finish
        startrun = int(re.match('[0-9]*$', rundir).groups()) + 1
    else:
        startrun = 0
        #with open(args.outdir / 'config.json', 'w') as f:
        #    json.dump(vars(args), f)

    # collect all detected parameters in "p"
    p = attrdict({'outdir': args.outdir})

    setup_seed(args, p, log)

    p.update(process_newton_args(args, log))
    unimarg = None
    if p.bimarg is not None:
        p['L'], p['q'] = getLq(p.bimarg)
        unimarg = getUnimarg(p.bimarg)

    p.update(process_potts_args(args, p.L, p.q, unimarg, log))
    L, q, alpha = p.L, p.q, p.alpha

    p.update(process_sample_args(args, log))
    gpup = process_GPU_args(args, L, q, p.outdir, log)
    p.update(gpup)
    gpus = setup_GPUs(p, log)
    gpus.initMCMC(p.nsteps)
    gpus.initJstep()

    # first gpu/node may need to store all collected seqs
    if p.distribute_jstep == 'head_gpu':
        gpus.head_gpu.initLargeBufs(gpus.nwalkers)
    elif p.distribute_jstep == 'head_node':
        if not MPI:
            raise Exception('"head_node" option only makes sense when '
                            'using MPI')
        nlrg = divideWalkers(gpus.nwalkers, gpus.head_node.ngpus, log, p.wgsize)
        gpus.head_node.initLargeBufs(nlrg)
    else:  # all
        pass

    log("")
    
    unimarg = getUnimarg(p.bimarg)
    gen_indep = args.seqs == 'independent' or args.init_model == 'independent'
    if gen_indep or args.reseed == 'independent':
        gpus.prepare_indep(unimarg)

    # figure out how many sequences we need to initialize
    needed_seqs = None
    use_seed = p.reseed in ['single_best', 'single_random']
    # we only need seqs for preopt, (and for indep use GPU later)
    if (p.preopt or (p.reseed == 'none')) and not gen_indep:
        needed_seqs = gpus.nseq['main']
    p.update(process_sequence_args(args, L, alpha, log, unimarg,
                                   nseqs=needed_seqs, needseed=use_seed))
    if p.reseed == 'msa':
        seedseqs = loadSequenceFile(args.seedmsa, alpha, log)
        seedseqs = repeatseqs(seedseqs, gpus.nseq['main'])
        p['seedmsa'] = np.split(seedseqs, gpus.ngpus)

    # initialize main buffers with any given sequences
    if p.preopt:
        if p.reseed == 'none':
            if p.seqs is None:
                raise Exception("Need to provide seqs if not using seedseq")
            log("")
            log("Initializing main seq buf with loaded seqs.")
            gpus.setSeqs('main', p.seqs, log)
        elif gen_indep:
            log("Generating Indep seqs on GPUs")
            gpus.gen_indep('main')
            seqs = gpus.collect(['seq main'])[0]
            writeSeqs('test_seq', seqs)
        else:
            raise Exception("Need to specify initial seqs for preopt")
    elif use_seed and p.seedseq is None:
        raise Exception("Must provide seedseq if using reseed=single_*")

    log("")

    log("Computation Overview")
    log("====================")
    log("Running {} Newton-MCMC rounds".format(p.mcmcsteps))
    if p.equiltime == 'auto':
        log(("In each round, running {} MC walkers until equilibrated, with a "
             "minimum of {} equilibration loops").format(
              p.nwalkers, p.min_equil))
    else:
        log(("In each round, running {} MC walkers for {} equilibration loops "
             "with {} MC steps per loop (Each walker equilibrated a total of "
             "{} MC steps, or {:.1f} steps per position)."
             ).format(p.nwalkers, p.equiltime, p.nsteps, p.nsteps*p.equiltime,
                    p.nsteps*p.equiltime/p.L))

    describe_tempering(args, p, log)

    N = p.nwalkers
    if p.tempering:
        B0 = p.tempering[0]
        N = np.sum(p.tempering == B0)

    f = p.bimarg
    expected_SSR = np.sum(f*(1-f))/N
    absexp = np.sqrt(2/np.pi)*np.sqrt(f*(1-f)/N)/f
    expected_Ferr = np.mean(absexp[f>0.01])
    log("\nEstimated lowest achievable statistical error for this nwalkers and "
        "bimarg is:\nMIN:    SSR = {:.4f}   Ferr = {:.3f}".format(expected_SSR,
                                                                 expected_Ferr))
    log("(Statistical error only. Modeling biases and perturbation procedure "
        "may cause additional error)")

    log("")
    log("")
    log("MCMC Run")
    log("========")

    p['max_ns'] = 2048
    p['peak_ns'] = 256
    p['cur_ns'] = 256

    mi3gpu.NewtonSteps.newtonMCMC(p, gpus, startrun, log)

    if args.log:
        logfile.close()

def getEnergies(orig_args, args, log):
    descr = ('Compute Potts Energy of a set of sequences')
    parser = argparse.ArgumentParser(prog=progname + ' getEnergies',
                                     description=descr)
    add = parser.add_argument
    add('out', default='output', help='Output File')
    addopt(parser, 'GPU Options',         'wgsize gpus profile')
    addopt(parser, 'Potts Model Options', 'alpha couplings')
    addopt(parser, 'Sequence Options',    'seqs')
    addopt(parser,  None,                 'outdir log')

    #genenergies uses a subset of the full inverse ising parameters,
    #so use custom set of params here

    args = parser.parse_args(args)
    args.measurefperror = False

    requireargs(args, 'couplings alpha seqs')

    args.outdir.mkdir(parents=True)
    if args.log:
        logfile = open(args.outdir / 'log', 'wt')
        log = lambda *s, **kwds: print(*s, file=logfile, flush=True, **kwds)

    print_node_startup(log, orig_args)

    log("Initialization")
    log("===============")
    log("")

    p = attrdict({'outdir': args.outdir})
    p.update(process_potts_args(args, None, None, None, log))
    L, q, alpha = p.L, p.q, p.alpha
    log("Sequence Setup")
    log("--------------")
    seqs = loadSequenceFile(args.seqs, alpha, log)
    if seqs is None:
        raise Exception("seqs must be supplied")
    log("")

    args.nwalkers = len(seqs)
    args.nsteps = 1
    args.nlargebuf = 1
    gpup = process_GPU_args(args, L, q, p.outdir, log)
    p.update(gpup)
    gpus = setup_GPUs(p, log)
    gpus.setSeqs('main', seqs, log)
    log("")


    log("Computing Energies")
    log("==================")

    gpus.setBuf('J', p.couplings)
    gpus.calcEnergies('main')
    es = gpus.collect('E main')

    log("Saving results to file '{}'".format(args.out))
    np.save(args.out, es)

    if args.log:
        logfile.close()

def MCMCbenchmark(orig_args, args, log):
    descr = ('Benchmark MCMC generation on the GPU')
    parser = argparse.ArgumentParser(prog=progname + ' benchmark',
                                     description=descr)
    add = parser.add_argument
    add('--nloop', type=np.uint32, required=True,
        help="Number of kernel calls to benchmark")
    addopt(parser, 'GPU options',         'nwalkers nsteps wgsize '
                                          'gpus profile')
    addopt(parser, 'Sequence Options',    'seedseq seqs')
    addopt(parser, 'Potts Model Options', 'alpha couplings L')
    addopt(parser,  None,                 'init_model outdir rngseed log')

    args = parser.parse_args(args)
    nloop = args.nloop
    args.measurefperror = False

    if args.couplings is None:
        raise ValueError("--couplings is required")

    if args.nwalkers is None:
        raise ValueError("--nwalkers is required")

    args.outdir.mkdir(parents=True)
    if args.log:
        logfile = open(args.outdir / 'log', 'wt')
        log = lambda *s, **kwds: print(*s, file=logfile, flush=True, **kwds)

    print_node_startup(log, orig_args)

    log("Initialization")
    log("===============")
    log("")

    p = attrdict({'outdir': args.outdir})

    setup_seed(args, p, log)
    p.update(process_potts_args(args, p.L, p.q, None, log))
    L, q, alpha = p.L, p.q, p.alpha

    #args.nlargebuf = 1

    setup_seed(args, p, log)

    gpup = process_GPU_args(args, L, q, p.outdir, log)
    p.update(gpup)
    gpus = setup_GPUs(p, log)
    gpus.initMCMC(p.nsteps)

    # figure out how many sequences we need to initialize
    needed_seqs = None
    use_seed = False
    if args.seqs is not None:
        needed_seqs = gpus.nseq['main']
    elif args.seedseq is not None:
        use_seed = True
    else:
        raise Exception("'seqs' or 'seedseq' option required")
    unimarg = getUnimarg(p.bimarg)
    p.update(process_sequence_args(args, L, alpha, log, unimarg,
                                   nseqs=needed_seqs, needseed=use_seed))
    if p.reseed == 'msa':
        seedseqs = loadSequenceFile(args.seedmsa, alpha, log)
        seedseqs = repeatseqs(seedseqs, gpus.nseq['main'])
        p['seedmsa'] = np.split(seedseqs, gpus.ngpus)

    # initialize main buffers with any given sequences
    if use_seed:
        if p.seedseq is None:
            raise Exception("Must provide seedseq if using reseed=single_*")
        gpus.fillSeqs(p.seedseq)
    else:
        if p.seqs is None:
            raise Exception("Need to provide seqs if not using seedseq")
        gpus.setSeqs('main', p.seqs, log)

    log("")

    log("Benchmark")
    log("=========")
    log("")
    log("Benchmarking MCMC for {} loops, {} MC steps per loop".format(
                                                 nloop, p.nsteps))
    import time

    def runMCMC():
        for i in range(nloop):
            gpus.runMCMC()
        gpus.wait()

    #initialize
    gpus.setBuf('J', p.couplings)

    #warmup
    log("Warmup run...")
    runMCMC()

    #timed run
    log("Timed run...")
    start = time.perf_counter()
    runMCMC()
    end = time.perf_counter()

    log("Elapsed time: ", end - start)
    totsteps = p.nwalkers*nloop*np.float64(p.nsteps)
    steps_per_second = totsteps/(end-start)
    log("MC steps computed: {}".format(totsteps))
    log("MC steps per second: {:g}".format(steps_per_second))

    ## quick sanity check as a bonus
    #gpus.calcEnergies('main')
    #es = gpus.collect('E main')
    #log("\nConsistency check: <E> = ", np.mean(es), np.std(es))

    if args.log:
        logfile.close()

def equilibrate(orig_args, args, log):
    descr = ('Run a round of MCMC generation on the GPU')
    parser = argparse.ArgumentParser(prog=progname + ' mcmc',
                                     description=descr)
    add = parser.add_argument
    addopt(parser, 'GPU options',         'nwalkers nsteps wgsize '
                                          'gpus profile')
    addopt(parser, 'Sequence Options',    'seedseq seqs indep_marg ')
    addopt(parser, 'Sampling Options',    'equiltime min_equil '
                                          'trackequil tracked '
                                          'tempering nswaps_temp')
    addopt(parser, 'Potts Model Options', 'alpha couplings L')
    addopt(parser,  None,                 'init_model outdir log rngseed')

    args = parser.parse_args(args)
    args.measurefperror = False

    args.outdir.mkdir(parents=True)
    if args.log:
        logfile = open(args.outdir / 'log', 'wt')
        log = lambda *s, **kwds: print(*s, file=logfile, flush=True, **kwds)

    print_node_startup(log, orig_args)

    log("Initialization")
    log("===============")

    p = attrdict({'outdir': args.outdir})

    setup_seed(args, p, log)

    p.update(process_potts_args(args, None, None, None, log))
    L, q, alpha = p.L, p.q, p.alpha

    p.update(process_sample_args(args, log))
    if p.equiltime == 'auto':
        rngPeriod = 0
    else:
        rngPeriod = p.equiltime

    gpup = process_GPU_args(args, L, q, p.outdir, log)
    p.update(gpup)
    gpus = setup_GPUs(p, log)
    gpus.initMCMC(p.nsteps)

    gen_indep = args.seqs == 'independent' or args.init_model == 'independent'
    if gen_indep:
        imarg = None
        if args.indep_marg is not None:
            imarg = np.load(args.indep_marg)
        elif args.init_model:
            fn = Path(args.init_model, 'bimarg.npy')
            if fn.is_file():
                imarg = np.load(fn)
        if imarg is None:
            raise ValueError("indep_marg must be supplied if generating "
                             "independent-model sequences")
        if imarg.shape == (L, q):
            log("loading unimarg from {} for independent model sequence "
                "generation".format(args.indep_marg))
            unimarg = imarg
        else:
            log("loading bimarg from {} and converted to unimarg for "
                "independent model sequence generation".format(args.indep_marg))
            unimarg = getUnimarg(imarg)
        gpus.prepare_indep(unimarg)

    nseqs = None
    needseed = False
    if args.seqs is not None:
        nseqs = gpus.nseq['main']
    if args.seedseq is not None:
        needseed = True
    elif args.seqs != 'independent':
        nseqs = p.nwalkers
    p.update(process_sequence_args(args, L, alpha, log, nseqs=nseqs,
                                   needseed=needseed))
    log("")

    log("Computation Overview")
    log("====================")
    if p.equiltime == 'auto':
        log(("In each round, running {} MC walkers until equilibrated, with a "
             "minimum of {} equilibration loops").format(
              p.nwalkers, p.min_equil))
    else:
        log(("Running {} MC walkers for {} equilibration loops "
             "with {} MC steps per loop (Each walker equilibrated a total of "
             "{} MC steps, or {:.1f} steps per position)."
             ).format(p.nwalkers, p.equiltime, p.nsteps, p.nsteps*p.equiltime,
                      p.nsteps*p.equiltime/p.L))

    describe_tempering(args, p, log)

    # set up gpu buffers
    if needseed:
        gpus.fillSeqs(p.seedseq)
    elif args.seqs == 'independent':
        gpus.gen_indep('main')
    else:
        gpus.setSeqs('main', p.seqs, log)

    gpus.setBuf('J', p.couplings)

    log("")

    log("Equilibrating")
    log("====================")

    MCMC_func = mi3gpu.NewtonSteps.runMCMC

    # set up tempering if needed
    if p.tempering is not None:
        MCMC_func = mi3gpu.NewtonSteps.runMCMC_tempered
        B0 = np.max(p.tempering)

        if p.nwalkers % len(p.tempering) != 0:
            raise Exception("# of temperatures must evenly divide # walkers")
        Bs = np.concatenate([full(p.nwalkers/len(p.tempering), b, dtype='f4')
                          for b in p.tempering])
        Bs = split(Bs, len(gpus))
        for B,gpu in zip(Bs, gpus):
            gpu.setBuf('Bs', B)
            gpu.markSeqs(B == B0)

    (bimarg_model,
     bicount,
     sampledenergies,
     e_rho,
     ptinfo,
     equilsteps) = MCMC_func(gpus, p.couplings, 'gen', p, log)

    seqs = gpus.collect('seq main')

    outdir = p.outdir
    np.savetxt(outdir / 'bicounts', bicount, fmt='%d')
    np.save(outdir / 'bimarg', bimarg_model)
    np.save(outdir / 'energies', sampledenergies)
    writeSeqs(outdir / 'seqs', seqs, alpha)

    if p.tempering is not None:
        e, b = readGPUbufs(['E main', 'Bs'], gpus)
        np.save(outdir / 'walker_Bs', np.concatenate(b))
        np.save(outdir / 'walker_Es', np.concatenate(e))
        log("Final PT swap rate: {}".format(ptinfo[1]))

    log("Mean energy:", np.mean(sampledenergies))

    log("Done!")

    if args.log:
        logfile.close()

def subseqFreq(orig_args, args, log):
    descr = ('Compute relative frequency of subsequences at fixed positions')
    parser = argparse.ArgumentParser(prog=progname + ' subseqFreq',
                                     description=descr)
    add = parser.add_argument
    add('fixpos', help="comma separated list of fixed positions")
    add('out', default='output', help='Output File')
    addopt(parser, 'GPU options',         'wgsize gpus')
    addopt(parser, 'Potts Model Options', 'alpha couplings L')
    addopt(parser,  None,                 'outdir log')
    group = parser.add_argument_group('Sequence Options')
    add = group.add_argument
    add('backgroundseqs', help="large sample of equilibrated sequences")
    add('subseqs', help="sequences from which to compute subseq freqs")

    args = parser.parse_args(args)
    args.measurefperror = False

    args.outdir.mkdir(parents=True)
    if args.log:
        logfile = open(args.outdir / 'log', 'wt')
        log = lambda *s, **kwds: print(*s, file=logfile, flush=True, **kwds)

    print_node_startup(log, orig_args)

    log("Initialization")
    log("===============")
    log("")

    p = attrdict({'outdir': args.outdir})
    args.trackequil = 0
    p.update(process_potts_args(args, p.L, p.q, None, log))
    L, q, alpha = p.L, p.q, p.alpha

    # try to load sequence files
    bseqs = loadSequenceFile(args.backgroundseqs, alpha, log)
    sseqs = loadSequenceFile(args.subseqs, alpha, log)

    args.nwalkers = 1
    gpup, cldat, gdevs = process_GPU_args(args, L, q, p.outdir, 1, log)
    p.update(gpup)
    p.nsteps = 1
    gpuwalkers = divideWalkers(len(bseqs), len(gdevs), log, p.wgsize)
    gpus = [initGPU(n, cldat, dev, len(sseqs), nwalk, p, log)
            for n,(dev, nwalk) in enumerate(zip(gdevs, gpuwalkers))]

    #fix positions
    fixedpos = np.array([int(x) for x in args.fixpos.split(',')])
    fixedmarks = np.zeros(L, dtype='u1')
    fixedmarks[fixedpos] = 1

    #load buffers
    gpubseqs = split(bseqs, np.cumsum(gpuwalkers)[:-1])
    for gpu,bs in zip(gpus, gpubseqs):
        gpu.setBuf('seq main', sseqs)
        gpu.setBuf('seq large', bs)
        gpu.markPos(fixedmarks)
        gpu.setBuf('J', p.couplings)

    log("")

    log("Subsequence Frequency Calculation")
    log("=================================")
    log("")

    for gpu in gpus:
        gpu.calcEnergies('large')
    origEs = np.concatenate(readGPUbufs(['E large'], gpus)[0])

    log("Getting substituted energies...")
    subseqE = []
    logf = np.zeros(len(sseqs))
    for n in range(len(sseqs)):
        # replaced fixed positions by subsequence, and calc energies
        for gpu in gpus:
            gpu.copySubseq(n)
            gpu.calcEnergies('large')
        energies = np.concatenate(readGPUbufs(['E large'], gpus)[0])
        logf[n] = logsumexp(origEs - energies)

    #save result
    log("Saving result (log frequency) to file {}".format(args.out))
    np.save(args.out, logf)

    if args.log:
        logfile.close()

################################################################################

def process_GPU_args(args, L, q, outdir, log):
    log("GPU setup")
    log("---------")

    param = attrdict({'nsteps': args.nsteps,
                      'wgsize': args.wgsize,
                      'nwalkers': args.nwalkers,
                      'gpuspec': args.gpus,
                      'profile': args.profile,
                      'fperror': args.measurefperror})

    p = attrdict(param.copy())
    p.update({'L': L, 'q': q, 'outdir': outdir})

    p['wgsize'] = wgsize_heuristic(p.q, p.wgsize)

    log("Total GPU walkers: {}".format(p.nwalkers))
    log("Work Group Size: {}".format(p.wgsize))
    log("{} MC steps per MCMC kernel call".format(p.nsteps))
    if p.profile:
        log("Profiling Enabled")
    return p

def process_newton_args(args, log):
    log("Newton Solver Setup")
    log("-------------------")
    mcmcsteps = args.mcsteps
    log("Running {} Newton-MCMC rounds".format(mcmcsteps))

    param = {'mcmcsteps': args.mcsteps,
             'newtonSteps': args.newtonsteps,
             'newton_delta': args.newton_delta,
             'fracNeff': args.fracNeff,
             'gamma0': args.gamma,
             'pcdamping': args.damping,
             'reseed': args.reseed,
             'preopt': args.preopt,
             'distribute_jstep': args.distribute_jstep}

    p = attrdict(param)

    log("Updating J locally with gamma={}, and pc-damping {}".format(
        str(p.gamma0), str(p.pcdamping)))
    log("Running {} Newton update steps per round.".format(p.newtonSteps))
    log("Using {}-GPU mode for Newton-step calculations.".format(
        p.distribute_jstep))

    log("Reading target marginals from file {}".format(args.bimarg))
    bimarg = np.load(args.bimarg)
    if bimarg.dtype != np.dtype('<f4'):
        raise Exception("Bimarg must be in 'f4' format")
        #could convert, but this helps warn that something may be wrong
    if np.any((bimarg <= 0) | (bimarg > 1)):
        raise Exception("All bimarg must be 0 < f < 1")
    log("Target Marginals: " + printsome(bimarg) + "...")
    p['bimarg'] = bimarg

    if args.reg is not None:
        rtype, dummy, rarg = args.reg.partition(':')
        rtypes = ['l2z', 'l1z', 'SCADX', 'X', 'Xij', 'ddE']
        if rtype not in rtypes:
            raise Exception("reg must be one of {}".format(str(rtypes)))
        p['reg'] = rtype
        if rtype == 'ddE':
            lam = float(rarg)
            log("Regularizing using ddE with lambda = {}".format(lam))
            p['regarg'] = (lam,)
        elif rtype == 'l2z' or rtype == 'l1z':
            try:
                lJ = float(rarg)
                log(("Regularizing using {} norm with lambda_J = {}").format(
                                                                     rtype, lJ))
            except:
                raise Exception("{r} specifier must be of form '{r}:lJ', eg "
                          "'{r}:0.01'. Got '{}'".format(args.reg, r=rtype))
            p['regarg'] = (lJ,)
        elif rtype == 'X':
            try:
                lX = float(rarg)
                log(("Regularizing X with lambda_X = {}").format(lX))
            except:
                raise Exception("{r} specifier must be of form 'X:lX', eg "
                          "'X:0.01'. Got '{}'".format(args.reg, r=rtype))
            p['regarg'] = (lX,)
        elif rtype == 'Xij':
            log("Regularizing with Xij from file {}".format(rarg))
            p['regarg'] = np.load(rarg)
            if p['regarg'].shape != bimarg.shape:
                raise Exception("Xij in wrong format")
        elif rtype == 'SCADX':
            try:
                d, dummy, r = rarg.partition(':')
                r, dummy, a = r.partition(':')
                d = float(d)
                r = float(r)
                a = float(a) if a != '' else 4.0
                if a < 2.0:
                    raise Exception("SCADX a parameter must be >= 2.0")
                s = 2*d/((1+a)*r*r) # see comment in mcmc.cl
                log(("Regularizing using SCADX with d={} r={} a={}"
                    ).format(d, r, a))
            except:
                raise Exception("{r} specifier must be of form '{r}:d:r:a' or "
                              "'{r}:d:r', eg '{r}:10:0.1'. Got '{arg}'".format(
                                arg=args.reg))
            p['regarg'] = (s, r, a)

    log("")
    return p

def updateLq(L, q, newL, newq, name):
    # update L and q with new values, checking that they
    # are the same as the old values if not None
    if newL is not None:
        if L is not None and L != newL:
            raise Exception("L from {} ({}) inconsitent with previous "
                            "value ({})".format(name, newL, L))
        L = newL
    if newq is not None:
        if q is not None and q != newq:
            raise Exception("q from {} ({}) inconsitent with previous "
                            "value ({})".format(name, newq, q))
        q = newq
    return L, q

def process_potts_args(args, L, q, unimarg, log):
    log("Potts Model Setup")
    log("-----------------")

    # we try to infer L and q from any values given. The possible sources
    # * command line options -L and -q
    # * from bivariate_target dimensions
    # * from coupling dimensions

    alpha = args.alpha.strip()
    argL = args.L if hasattr(args, 'L') else None
    L, q = updateLq(argL, len(alpha), L, q, 'bimarg')

    # next try to get couplings (may determine L, q)
    couplings, L, q = getCouplings(args, L, q, unimarg, log)
    # we should have L and q by this point

    log("alphabet: {}".format(alpha))
    log("q {}  L {}".format(q, L))
    log("Couplings: " + printsome(couplings) + "...")

    log("")
    return attrdict({'L': L, 'q': q, 'alpha': alpha,
                     'couplings': couplings})

def getCouplings(args, L, q, unimarg, log):
    couplings = None

    if args.couplings is None and args.init_model in ['uniform', 'independent']:
        args.couplings = args.init_model

    if args.couplings:
        #first try to generate couplings (requires L, q)
        if args.couplings in ['uniform', 'independent']:
            if L is None: # we are sure to have q
                raise Exception("Need L to generate couplings")
        if args.couplings == 'uniform':
            log("Setting Initial couplings to uniform frequencies")
            h = -np.log(1.0/q)
            J = np.zeros((L*(L-1)//2,q*q), dtype='<f4')
            couplings = fieldlessGaugeEven(h, J)[1]
        elif args.couplings == 'independent':
            log("Setting Initial couplings to independent model")
            if unimarg is None:
                raise Exception("Need univariate marginals to generate "
                                "independent model couplings")
            h = -np.log(unimarg)
            J = np.zeros((L*(L-1)//2,q*q), dtype='<f4')
            couplings = fieldlessGaugeEven(h, J)[1]
        else: #otherwise load them from file
            log("Reading couplings from file {}".format(args.couplings))
            couplings = np.load(args.couplings)
            if couplings.dtype != np.dtype('<f4'):
                raise Exception("Couplings must be in 'f4' format")
    elif args.init_model and args.init_model not in ['uniform', 'independent']:
        # and otherwise try to load them from model directory
        fn = Path(args.init_model, 'J.npy')
        if fn.is_file():
            log("Reading couplings from file {}".format(fn))
            couplings = np.load(fn)
            if couplings.dtype != np.dtype('<f4'):
                raise Exception("Couplings must be in 'f4' format")
        else:
            raise Exception("could not find file {}".format(fn))
    else:
        raise Exception("didn't get couplings or init_model")
    L2, q2 = getLq(couplings)
    L, q = updateLq(L, q, L2, q2, 'couplings')

    if couplings is None:
        raise Exception("Could not find couplings. Use either the "
                        "'couplings' or 'init_model' options.")

    return couplings, L, q

def repeatseqs(seqs, n):
    return np.repeat(seqs, (n-1)//seqs.shape[0] + 1, axis=0)[:n,:]

def process_sequence_args(args, L, alpha, log, unimarg=None,
                          nseqs=None, needseed=False):
    if (not nseqs) and (not needseed):
        return {}

    log("Sequence Setup")
    log("--------------")

    q = len(alpha)
    seedseq, seqs = None, None

    # try to load sequence files
    if nseqs is not None:
        if args.seqs in ['uniform', 'independent']:
            seqs = generateSequences(args.seqs, L, q, nseqs, log, unimarg)
        elif args.init_model in ['uniform', 'independent']:
            seqs = generateSequences(args.init_model, L, q, nseqs, log, unimarg)
        elif args.seqs is not None:
            seqs = loadSequenceFile(args.seqs, alpha, log)
        elif args.init_model is not None:
            seqs = loadSequenceDir(args.init_model, '', alpha, log)

        if nseqs is not None and seqs is None:
            raise Exception("Did not find requested {} sequences".format(nseqs))

        n_loaded = seqs.shape[0]
        if nseqs > n_loaded:
            log("Repeating {} sequences to make {}".format(n_loaded, nseqs))
            seqs = repeatseqs(seqs, nseqs)
        elif nseqs < n_loaded:
            log("Truncating {} sequences to make {}".format( n_loaded, nseqs))
            seqs = seqs[:nseqs]

    # try to get seed seq
    if needseed:
        if args.seedseq in ['uniform', 'independent']:
            seedseq = generateSequences(args.seedseq, L, q, 1, log, unimarg)[0]
            seedseq_origin = args.seedseq
        elif args.seedseq is not None: # given string
            try:
                seedseq = np.array([alpha.index.index(c) for c in args.seedseq],
                                   dtype='<u1')
                seedseq_origin = 'supplied'
            except:
                seedseq = loadseedseq(args.seedseq, args.alpha.strip(), log)
                seedseq_origin = 'from file'
        elif args.init_model in ['uniform', 'independent']:
            seedseq = generateSequences(args.init_model, L, q, 1, unimarg, 
                                        log)[0]
            seedseq_origin = args.init_model
        elif args.init_model is not None:
            seedseq = loadseedseq(Path(args.init_model, 'seedseq'),
                                  args.alpha.strip(), log)
            seedseq_origin = 'from file'

        log("Seed seq ({}): {}".format(seedseq_origin,
                                       "".join(alpha[x] for x in seedseq)))

    log("")
    return attrdict({'seedseq': seedseq,
                     'seqs': seqs})

def generateSequences(gentype, L, q, nseqs, log, unimarg=None):
    if gentype == 'zero' or gentype == 'uniform':
        log("Generating {} random sequences...".format(nseqs))
        return randint(0, q, size=(nseqs, L)).astype('<u1')
    elif gentype == 'independent':
        log("Generating {} independent-model sequences...".format(nseqs))
        if unimarg is None:
            raise Exception("marg must be provided to generate sequences")
        cumprob = np.cumsum(unimarg, axis=1)
        cumprob = cumprob/(cumprob[:,-1][:,None]) #correct fp errors?
        return np.array([np.searchsorted(cp, rand(nseqs)) for cp in cumprob],
                     dtype='<u1').T
    raise Exception("Unknown sequence generation mode '{}'".format(gentype))

def loadseedseq(fn, alpha, log):
    log("Reading seedseq from file {}".format(fn))
    with open(fn) as f:
        seedseq = f.readline().strip()
        seedseq = np.array([alpha.index(c) for c in seedseq], dtype='<u1')
    return seedseq

def loadSequenceFile(sfile, alpha, log):
    log("Loading sequences from file {}".format(sfile))
    seqs = loadSeqs(sfile, alpha=alpha)[0].astype('<u1')
    log("Found {} sequences".format(seqs.shape[0]))
    return seqs

def loadSequenceDir(sdir, bufname, alpha, log):
    log("Loading {} sequences from dir {}".format(bufname, sdir))
    sfile = Path(sdir) / 'seqs'
    seqs = loadSeqs(sfile, alpha=alpha)[0].astype('<u1')
    log("Found {} sequences".format(seqs.shape[0]))
    return seqs

def process_sample_args(args, log):
    p = attrdict({'equiltime': args.equiltime,
                  'min_equil': args.min_equil,
                  'trackequil': args.trackequil,
                  'tracked': args.tracked.split(',')})
    
    bad_track = [x for x in p.tracked if x not in ['bim', 'E', 'seq']]
    if len(bad_track) != 0:
        raise ValueError('Invalid "--tracked" values: {}'.format(bad_track))

    if p['equiltime'] != 'auto':
        p['equiltime'] = int(p['equiltime'])


    if 'tempering' in args and args.tempering:
        try:
            Bs = np.load(args.tempering)
        except:
            Bs = np.array([x for x in args.tempering.split(",")], dtype='f4')
        p['tempering'] = Bs
        p['nswaps'] = args.nswaps_temp

    log("MCMC Sampling Setup")
    log("-------------------")

    if p.equiltime == 'auto':
        log('Using "auto" equilibration')
    else:
        log(("In each MCMC round, running {} GPU MCMC kernel calls"
             ).format(p.equiltime))
    if 'tempering' in p:
        log("Parallel tempering with inverse temperatures {}, "
            "swapping {} times per loop".format(args.tempering, p.nswaps))

    if p.equiltime != 'auto' and p.trackequil != 0:
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

def main(args):
    actions = {
      'infer':   inverseIsing,
      'energies':    getEnergies,
      'benchmark':   MCMCbenchmark,
      'subseq':      subseqFreq,
      'gen':         equilibrate,
     }

    descr = 'Perform biophysical Potts Model calculations on the GPU'
    parser = argparse.ArgumentParser(description=descr, add_help=False)
    add = parser.add_argument
    add('action', choices=actions.keys(), nargs='?', default=None,
        help="Computation to run")
    add('--clinfo', action=CLInfoAction, help="Display detected GPUs")
    add('--mpi', action='store_true', help="Enable MPI")
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

    if known_args.mpi:
        setup_MPI()

        if mpi_rank != 0:
            worker_GPU_main()
            return

    actions[known_args.action](args, remaining_args, print)

if __name__ == '__main__':
    setup_exit_hook(print)
    main(sys.argv[1:])

#!/usr/bin/env python2
from scipy import *
import scipy
from numpy.random import randint
import numpy.random
import pyopencl as cl
import sys, os, errno, glob, argparse, time
import load
from scipy.optimize import leastsq

numpy.random.seed(1234) #comment this if you don't want identical runs

#Recommmended/Example command line:
#stdbuf -i0 -o0 -e0 ./mcmcGPU.py bimarg.npy 0.0001 100 16384 229 ABCDEFGH rand -couplings logscore -preopt logscore -o outdir >log &

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
printsome = lambda a: " ".join(map(str,a.flatten()[:5]))

os.environ['PYOPENCL_COMPILER_OUTPUT'] = '0'
os.environ['PYOPENCL_NO_CACHE'] = '1'
os.environ["CUDA_CACHE_DISABLE"] = '1'

def getCouplingMatrix(couplings):
    coupleinds = [(a,b) for a in range(L-1) for b in range(a+1, L)]
    C = empty((L,nB,L,nB))*nan
    for n,(i,j) in enumerate(coupleinds): 
        block = couplings[n].reshape(nB,nB)
        C[i,:,j,:] = block
        C[j,:,i,:] = block.T
    return C

def zeroGauge(Js): #convert to zero gauge
    #surely this can be faster
    Jx = Js.reshape((nPairs, nB, nB))
    JxC = nan_to_num(getCouplingMatrix(Js))

    J0 = (Jx - mean(Jx, axis=1)[:,newaxis,:] 
             - mean(Jx, axis=2)[:,:,newaxis] 
             + mean(Jx, axis=(1,2))[:,newaxis,newaxis])
    h0 = sum(mean(JxC, axis=1), axis=0)
    h0 = h0 - mean(h0, axis=1)[:,newaxis]
    J0 = J0.reshape((J0.shape[0], nB**2))
    return h0, J0

def zeroJGauge(Js): #convert to zero gauge
    #surely this can be faster
    Jx = Js.reshape((nPairs, nB, nB))

    J0 = (Jx - mean(Jx, axis=1)[:,newaxis,:] 
             - mean(Jx, axis=2)[:,:,newaxis] 
             + mean(Jx, axis=(1,2))[:,newaxis,newaxis])

    JxC = nan_to_num(getCouplingMatrix(Js))
    h0 = sum(mean(JxC, axis=1), axis=0) - (sum(mean(JxC, axis=(1,3)), axis=0)/2)[:,newaxis]
    J0 = J0.reshape((J0.shape[0], nB**2))
    return h0, J0

def fieldlessGauge(hs, Js): #convert to a fieldless gauge
    #note: Fieldless gauge is not fully constrained: There
    #are many possible choices that are fieldless, this just returns one of them
    #This function tries to distribute the fields as evenly as possible
    J0 = Js.copy()
    hd = hs/(L-1)
    for n,(i,j) in enumerate([(i,j) for i in range(L-1) for j in range(i+1,L)]):
        J0[n,:] += repeat(hd[i,:], nB)
        J0[n,:] += tile(hd[j,:], nB)
    return J0

def getEnergies(s, couplings): #superseded by kernel function?
    pairenergy = zeros(s.shape[0], dtype='<f4')
    for n,(i,j) in enumerate([(i,j) for i in range(L-1) for j in range(i+1,L)]):
        pairenergy += couplings[n,nB*s[:,i] + s[:,j]]
    return pairenergy

################################################################################
# Read in args and Data:

print "Initialization\n==============="

parser = argparse.ArgumentParser(description='GD-MCMC')
parser.add_argument('bimarg')
parser.add_argument('gamma2', type=float32)
parser.add_argument('gdsteps', type=uint32)
parser.add_argument('nloop', type=uint32)
parser.add_argument('nsteps', type=uint32)
parser.add_argument('nsampleloops', type=uint32)
parser.add_argument('nsamples', type=uint32)
parser.add_argument('alpha', help="Alphabet, a sequence of letters")
parser.add_argument('startseq', help="Either a sequence, or 'rand'") 
parser.add_argument('-couplings', default='zero', 
                    help="One of 'zero', 'logscore', or a filename")
parser.add_argument('-outdir', default='output')
parser.add_argument('-pc', default=0)
parser.add_argument('-pcdamping', default=0.001)
parser.add_argument('-Jcutoff', default=None)
parser.add_argument('-preopt', default='none', 
                    help="One of 'rand', 'logscore', or a filename")

args = parser.parse_args(sys.argv[1:])

outdir = args.outdir
mkdir_p(outdir)

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

                        # for example:
gamma2 = args.gamma2    # 0.001
gdsteps = args.gdsteps  # 10
nloop = args.nloop      # 100
nsteps = args.nsteps    # 100 #make this a multiple of L, and ~ 100 - 500
nsampleloops = args.nsampleloops    
nsamples = args.nsamples  

if nsamples == 0:
    raise Exception("nsamples must be at least 1")

if nsteps%L != 0:
    print "WARNING: nsteps is not a multiple of L. This is probably WRONG."
    #if nsteps is not > L the MCMC generation will not vary
    #all parts of the sequences. If not a multiple, different parts
    #get equilibrated by different amounts

WGSIZE = 256 #tweak me. OpenCL work group size for MCMC kernel.
NGROUPS = 256 #number of work groups for MCMC kernel
NSEQ = WGSIZE*NGROUPS #number of seqs simulated
VSIZE = 256 #as large as possible, multiple of 2. Sumweights kernel has a float array of this size.
CGSIZE = 1024 #as large as possible
#sequences are padded to 32 bits
SWORDS = ((L-1)/4+1)  #number of words needed to store a sequence
SBYTES = (4*SWORDS)   #number of bytes needed to store a sequence
NHIST = 64 #must be a power of two. Number of histograms used in counting 
           #kernels (each hist is nB*nB floats/uints). Should really be
           # made a function of nB, eg 4096/(nB*nB) for occupancy ~ 3

Jcutoff = args.Jcutoff

print ("Running {} loops then sampling every {} loops to get {} samples ({} total seqs) "
       "with {} MC steps per loop, for {} GD steps.").format(
       nloop, nsampleloops, nsamples, nsamples*NSEQ, nsteps, gdsteps)
print "Updating J locally with gamma2 = {}, and dJ cutoff {}".format(gamma2, Jcutoff)

bicount = empty((nPairs, nB*nB), dtype='<u4')
alpha = args.alpha
if len(alpha) != nB:
    print "Expected alphabet size {}, got {}".format(nB, len(alpha))
    exit()

if args.couplings == 'zero':
    print "Setting Initial couplings to 0"
    couplings = zeros((nPairs, nB*nB), dtype='<f4')
elif args.couplings == 'logscore':
    print "Setting Initial couplings to Independent Log Scores"
    ff = bimarg_target.reshape((nPairs,nB,nB))
    marg = array([sum(ff[0],axis=1)] + [sum(ff[n],axis=0) for n in range(L-1)])
    marg = marg/(sum(marg,axis=1)[:,newaxis]) # correct any fp errors
    h = -log(marg)
    h = h - mean(h, axis=1)[:,newaxis]
    couplings = fieldlessGauge(h, zeros((nPairs,nB*nB),dtype='<f4'))
else:
    print "Reading initial couplings from file {}".format(args.couplings)
    couplings = scipy.load(args.couplings)
    if couplings.dtype != dtype('<f4'):
        raise Exception("Couplings in wrong format")
#switch to 'even' fieldless gauge for nicer output
h0, J0 = zeroJGauge(couplings)
couplings = fieldlessGauge(h0, J0)
save(os.path.join(outdir, 'startJ'), couplings)

if args.startseq == 'rand':
    startseq = randint(0, nB, size=L).astype('<u1')
else:
    startseq = array([alpha.index(c) for c in args.startseq], dtype='<u1')
sstype = 'random' if args.startseq == 'rand' else 'provided'

print "Target Marginals: " + printsome(bimarg_target) + "..."
print "Initial Couplings: " + printsome(couplings) + "..."
print "Start seq ({}): {}".format(sstype, "".join(alpha[x] for x in startseq))

################################################################################
#Initialize GPU

print ""
platforms = cl.get_platforms()
for n,p in enumerate(platforms):
    print "Platform {} '{}'{}:".format(n, p.name, " (Chosen)" if n == 0 else "")
    print "    Vendor: {}".format(p.vendor)
    print "    Version: {}".format(p.version)
    print "    Extensions: {}".format(p.extensions)
print ""
devices = platforms[0].get_devices()
for n,d in enumerate(devices):
    print "Device {} '{}'{}:".format(n, d.name, " (Chosen)" if n == 0 else "" )
    print "    Vendor: {}".format(d.vendor)
    print "    Version: {}".format(d.version)
    print "    Driver Version: {}".format(d.driver_version)
    print "    Max Clock Frequency: {}".format(d.max_clock_frequency)
    print "    Max Compute Units: {}".format(d.max_compute_units)
    print "    Max Work Group Size: {}".format(d.max_work_group_size)
    print "    Global Mem Size: {}".format(d.global_mem_size)
    print "    Global Mem Cache Size: {}".format(d.global_mem_cache_size)
    print "    Local Mem Size: {}".format(d.local_mem_size)
    print "    Max Constant Buffer Size: {}".format(d.max_constant_buffer_size)
print ""

print "Getting CL Context"
ctx = cl.Context([devices[0]])
print "Getting CL Queue"
queue = cl.CommandQueue(ctx)
print "Loading CL src"

srcfn = "metropolis.cl"
with open(os.path.join(scriptPath, srcfn)) as f:
    src = f.read()
options = [('NGROUPS', NGROUPS), ('WGSIZE', WGSIZE), ('NSEQS', NSEQ), 
           ('VSIZE', VSIZE), ('NHIST', NHIST), ('CGSIZE', CGSIZE),
           ('SEED', randint(2**32)), ('nB', nB), ('L', L), ('PC', args.pcdamping),
           ('nsteps', nsteps)]
if Jcutoff:
    options.append(('JCUTOFF', Jcutoff))
optstr = " ".join(["-D {}={}".format(opt,val) for opt,val in options]) 
print "Compilation Options: ", optstr
extraopt = " -cl-nv-verbose -Werror -I {}".format(scriptPath)
print "Compiling CL..."
prg = cl.Program(ctx, src).build(optstr + extraopt)
print "\nOpenCL Compilation Log:"
print prg.get_build_info(devices[0], cl.program_build_info.LOG)

#dump compiled program
ptx = prg.get_info(cl.program_info.BINARIES)
for n,p in enumerate(ptx):
    print "PTX length: ", len(p) #useful to see if compilation changed
    with open(os.path.join(outdir, 'ptx{}'.format(n)), 'wt') as f:
        f.write(p)

#arrange seq memory for coalesced access

#converts seqs to uchars, padded to 32bits, assume GPU is little endian
def packSeqs(seqs):
    bseqs = zeros((seqs.shape[0], SBYTES), dtype='<u1', order='C')
    bseqs[:,:L] = seqs  
    mem = zeros((SWORDS, seqs.shape[0]), dtype='<u4', order='C')
    for i in range(SWORDS):
        mem[i,:] = bseqs.view(uint32)[:,i]
    return mem
def unpackSeqs(mem):
    bseqs = zeros((mem.shape[1], SBYTES), dtype='<u1', order='C')
    for i in range(SWORDS): #undo memory rearrangement
        bseqs.view(uint32)[:,i] = mem[i,:] 
    return bseqs[:,:L]

#convert from format where every row is a unique ij pair (L choose 2 rows)
#to format with every pair, all orders (L^2 rows)
#Note that the GPU kernel packfV does the same thing
def packCouplings(couplings):
    fullcouplings = zeros((L*L,nB*nB), dtype='<f4', order='C')
    pairs = [(i,j) for i in range(L-1) for j in range(i+1,L)]
    for n,(i,j) in enumerate(pairs):
        fullcouplings[L*i + j,:] = couplings[n,:]
        fullcouplings[L*j + i,:] = couplings[n,:].reshape((nB,nB)).T.flatten()
    return fullcouplings

bimarg_model = zeros(bimarg_target.shape, dtype='<f4')
bicount = zeros(bimarg_target.shape, dtype='<u4')
J = zeros(couplings.shape, dtype='<f4')
energies = zeros(WGSIZE*NGROUPS, dtype='<f4')
weights = zeros(WGSIZE*NGROUPS, dtype='<f4')
sampledenergies = empty(nsamples*NSEQ, dtype='<f4')
tmpv = zeros(n_couplings, dtype='<f4')
tmpJ = zeros(couplings.shape, dtype='<f4')

#allocate & init device memory
mf = cl.mem_flags
J_dev = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=couplings)
dJ_dev = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=couplings)
J_tmp_dev = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=couplings)
J_tmp_dev2 = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=couplings)
J_full_dev = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=packCouplings(couplings))
run_seed_dev = cl.Buffer(ctx, mf.READ_WRITE, size=4)
nseq_dev = cl.Buffer(ctx, mf.READ_WRITE, size=4)
offset_dev = cl.Buffer(ctx, mf.READ_WRITE, size=4)
energies_dev = cl.Buffer(ctx, mf.READ_WRITE, size=4*NSEQ)
energiessample_dev = cl.Buffer(ctx, mf.READ_WRITE, size=nsamples*4*NSEQ)
weights_dev = cl.Buffer(ctx, mf.READ_WRITE, size=4*nsamples*NSEQ)
seqmem_dev = cl.Buffer(ctx, mf.READ_WRITE, size=NSEQ*SBYTES)
seqsamples_dev = cl.Buffer(ctx, mf.READ_WRITE, size=nsamples*NSEQ*SBYTES)

bimarg_dev = cl.Buffer(ctx, mf.READ_WRITE, size=4*n_couplings)
bimarg_out_dev = cl.Buffer(ctx, mf.READ_WRITE, size=4*n_couplings)
bimarg_out_dev2 = cl.Buffer(ctx, mf.READ_WRITE, size=4*n_couplings)
bicount_dev = cl.Buffer(ctx, mf.READ_WRITE, size=4*n_couplings)
bimarg_target_dev = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=bimarg_target)
tmpv_dev = cl.Buffer(ctx, mf.READ_WRITE, size=4*n_couplings)

sumweights_dev = cl.Buffer(ctx, mf.READ_WRITE, size=4)

alpha_dev = cl.Buffer(ctx, mf.READ_WRITE, size=4)

################################################################################
#Functions which perform the computation

def writeStatus(n, rmsd, ssd, bicount, bimarg_model, couplings, seqs, startseq, energies):
    n = str(n)

    #print some details 
    disp = ["Start Seq: " + "".join([alpha[c] for c in startseq]),
            "RMSD: {}".format(rmsd),
            "SSD: {}".format(ssd),
            "Bicounts: " + printsome(bicount) + '...',
            "Marginals: " + printsome(bimarg_model) + '...',
            "Couplings: " + printsome(couplings) + "...",
            "Energies: Lowest =  {}, Mean = {}".format(min(energies), 
                                                       mean(energies))]
    dispstr = "\n".join(disp)
    
    mkdir_p(os.path.join(outdir, n))
    with open(os.path.join(outdir, n, 'info.txt'), 'wt') as f:
        print >>f, dispstr

    #save current state to file
    save(os.path.join(outdir, n, 'J'), couplings)
    savetxt(os.path.join(outdir, n, 'bicounts'), bicount, fmt='%d')
    save(os.path.join(outdir, n, 'bimarg'), bimarg_model)
    load.writeSites(os.path.join(outdir, n, 'finalseqs'), 
                    seqs, alpha, param={'alpha': alpha})

    print dispstr

def runMCMC(kernel_seed):
    cl.enqueue_copy(queue, run_seed_dev, kernel_seed)
    kernel_seed[0] += 1
    prg.metropolis(queue, (WGSIZE*NGROUPS,), (WGSIZE,), 
                          J_full_dev, run_seed_dev, seqmem_dev, energies_dev)

def countBimarg(bicount_dev, bimarg_dev, nseq, seqmem_dev):
    nseq = array([nseq], dtype='<u4')
    cl.enqueue_copy(queue, nseq_dev, nseq)
    chunksize = nPairs/(L-1)
    for i in range(0, nPairs, chunksize):
        offset = array([i], dtype='<u4')
        cl.enqueue_copy(queue, offset_dev, offset)
        prg.countBimarg(queue, (chunksize*NHIST,), (NHIST,), 
                      bicount_dev, bimarg_dev, offset_dev, nseq_dev, seqmem_dev)

def weightedMarg(bimarg_out_dev, weights_dev, sumweights_dev, nseq, seqmem_dev):
    cl.enqueue_copy(queue, nseq_dev, array([nseq], dtype='<u4'))
    chunksize = nPairs/(L-1)
    for i in range(0, nPairs, chunksize):
        cl.enqueue_copy(queue, offset_dev, array([i], dtype='<u4'))
        prg.weightedMarg(queue, (chunksize*NHIST,), (NHIST,),
                        bimarg_out_dev, weights_dev, sumweights_dev, 
                        offset_dev, nseq_dev, seqmem_dev)

def doFit(startseq, couplings):
    kernel_seed = array([0], dtype=int32)
    for i in range(gdsteps):
        rmsd, (seqs, energies) = singleStep('run_{}'.format(i), 
                                            kernel_seed, couplings, startseq)
        startseq = seqs[argmin(energies)]

def singleStep(runName, kernel_seed, couplings, startseq):
    print ""
    print "Gradient Descent step {}".format(runName)
    
    #run MCMC and read out seqs
    seqmem = packSeqs(tile(startseq, (NSEQ,1)))
    cl.enqueue_copy(queue, seqmem_dev, seqmem)
    cl.enqueue_copy(queue, J_dev, couplings)
    prg.packfV(queue, (nPairs*nB*nB,), (nB*nB,), J_dev, J_full_dev)
    #marginals = []
    for j in range(nloop):
        runMCMC(kernel_seed)
        #countBimarg(bicount_dev, bimarg_dev, NSEQ, seqmem_dev)
        #cl.enqueue_copy(queue, bimarg_model, bimarg_dev)
    #    marginals.append(bimarg_model.copy())
    #rr = [sum((marginals[-1] - m).flatten()**2) for m in marginals]
    #mkdir_p(os.path.join(outdir, runName))
    #savetxt(os.path.join(outdir, runName, 'mrun'), rr)

    cl.enqueue_copy(queue, seqmem, seqmem_dev)
    sampledseqs = [unpackSeqs(seqmem)]
    for j in range(nsamples-1):
        for k in range(nsampleloops):
            runMCMC(kernel_seed)
        cl.enqueue_copy(queue, seqmem, seqmem_dev)
        sampledseqs.append(unpackSeqs(seqmem))
    sampledseqs = concatenate(sampledseqs)

    #process sequences
    cl.enqueue_copy(queue, seqsamples_dev, packSeqs(sampledseqs))
    countBimarg(bicount_dev, bimarg_dev, nsamples*NSEQ, seqsamples_dev)
    prg.getEnergies(queue, (nsamples*NSEQ,), (WGSIZE,), 
                    J_full_dev, seqsamples_dev, energiessample_dev)

    #read out data from MCMC run
    cl.enqueue_copy(queue, bimarg_model, bimarg_dev)
    cl.enqueue_copy(queue, bicount, bicount_dev)
    rmsd = sqrt(mean((bimarg_target - bimarg_model)**2))
    ssd = sum((bimarg_target - bimarg_model)**2)
    cl.enqueue_copy(queue, sampledenergies, energiessample_dev)

    writeStatus(runName, rmsd, ssd, bicount, bimarg_model, 
                couplings, sampledseqs, startseq, sampledenergies)
    
    #compute new J using local optimization
    gradientDescent(128)
    cl.enqueue_copy(queue, couplings, J_dev)
    return rmsd, (sampledseqs, sampledenergies)

################################################################################
#local optimization related code

def perturbMarg(J_dev, bimarg_out_dev, nseq, seqmem_dev, energies_dev): 
    #overwrites J_full_dev, weights, sumweights
    #assumes seqmem_dev, energies_dev are filled in
    cl.enqueue_copy(queue, nseq_dev, array([nseq], dtype='<u4'))
    prg.packfV(queue, (nPairs*nB*nB,), (nB*nB,), J_dev, J_full_dev)
    prg.perturbedWeights(queue, (nseq,), (WGSIZE,), 
                   J_full_dev, nseq_dev, seqmem_dev, weights_dev, energies_dev)
    prg.sumWeights(queue, (VSIZE,), (VSIZE,), 
                   weights_dev, sumweights_dev, nseq_dev)
    weightedMarg(bimarg_out_dev, weights_dev, sumweights_dev, nseq, seqmem_dev)

def gradientDescent(niter):
    global couplings

    cl.enqueue_copy(queue, J_tmp_dev, couplings)
    cl.enqueue_copy(queue, J_dev, couplings)

    prg.packfV(queue, (nPairs*nB*nB,), (nB*nB,), J_tmp_dev, J_full_dev)
    prg.getEnergies(queue, (nsamples*NSEQ,), (WGSIZE,), 
                    J_full_dev, seqsamples_dev, energiessample_dev)

    alpha = array([gamma2], dtype='<f4') #step size
    cl.enqueue_copy(queue, alpha_dev, alpha)

    sumweights = array([0], dtype='<f4') 

    cl.enqueue_copy(queue, J_tmp_dev2, J_tmp_dev)
    cl.enqueue_copy(queue, bimarg_out_dev2, bimarg_out_dev)
    
    print "Local target: ", printsome(bimarg_target)
    print "Local optimization:"
    n = 1
    alphasteps = 16
    lastrmsd = inf
    for i in range(niter/alphasteps): 
        nrepeats = 0
        for k in range(alphasteps):
            perturbMarg(J_tmp_dev2, bimarg_out_dev2, nsamples*NSEQ, 
                        seqsamples_dev, energiessample_dev)

            #print out status
            cl.enqueue_copy(queue, bimarg_model, bimarg_out_dev2)
            cl.enqueue_copy(queue, sumweights, sumweights_dev)
            cl.enqueue_copy(queue, weights, weights_dev)
            rmsd = sum((bimarg_model.flatten() - bimarg_target.flatten())**2)
            print "{} r2: {} bimarg: {}   Neff: {:.1f} wspan: {:.3g} {:.2g}".format(n,rmsd,printsome(bimarg_model), sumweights[0], min(weights), max(weights))

            if rmsd > lastrmsd: #go back to last step
                alpha = alpha/2
                cl.enqueue_copy(queue, alpha_dev, alpha)
                nrepeats += 1
                print "Reducing alpha to {} and repeating step".format(alpha)
            else: #keep this step, and store current J and bm
                n += 1
                lastrmsd = rmsd
                cl.enqueue_copy(queue, J_tmp_dev, J_tmp_dev2)
                cl.enqueue_copy(queue, bimarg_out_dev, bimarg_out_dev2)

                #if sumweights[0] < nsamples*NSEQ/2 or max(weights) > 64:
                #if max(weights) > 64 or min(weights) < 1.0/64:
                #if max(weights)/min(weights) > 128:
                if max(weights) > 64:
                    print "Sequence weights diverging. Stopping"
                    cl.enqueue_copy(queue, J_dev, J_tmp_dev)
                    return
            
            prg.updatedJ(queue, (VSIZE,), (VSIZE,), 
                bimarg_target_dev, bimarg_out_dev, J_dev, alpha_dev, J_tmp_dev, J_tmp_dev2)

        if nrepeats == alphasteps:
            print "Too many rmsd inreases. Stopping"
            break

        alpha = alpha*2
        cl.enqueue_copy(queue, alpha_dev, alpha)
        print "Increasing alpha to {}".format(alpha)

    cl.enqueue_copy(queue, J_dev, J_tmp_dev)

################################################################################
#Run it!!!

print "\n\nMCMC Run\n========"

if args.preopt != 'none':
    if args.preopt == 'rand': #good for 'zero' couplings
        print "Pre-optimization (random sequences)"
        seqs = numpy.random.randint(0,nB,size=(nsamples*NSEQ, L)).astype('<u1')
    elif args.preopt == 'logscore': #good for 'logscore' (independent) couplings
        print "Pre-optimization (logscore independent sequences)"
        if args.couplings != 'logscore':
            raise Exception("Sorry, currently need logscore couplings "
                            "to use logscore preopt")
        cumprob = cumsum(marg, axis=1)
        cumprob = cumprob/(cumprob[:,-1][:,newaxis]) #correct fp errors?
        seqs = array([searchsorted(cp, rand(nsamples*NSEQ)) for cp in cumprob], 
                     dtype='<u1').T
    else:
        print "Pre-optimization (loading sequences from {})".format(args.preopt)
        print "(Warning: input couplings should have generated input sequences, or weird results ensue)"
        seqs = load.loadSites(args.preopt, names=alpha)[0].astype('<u1')
        if seqs.shape[0] != nsamples*NSEQ:
            raise Exception(("Error: Need {} preoptimization sequences, "
                             "got {}").format(nsamples*NSEQ, seqs.shape[0]))
    load.writeSites(os.path.join(outdir, 'preoptseqs'), 
                    seqs, alpha, param={'alpha': alpha})

    #initialize seqs, energies and marginals
    cl.enqueue_copy(queue, seqsamples_dev, packSeqs(seqs))
    
    cl.enqueue_copy(queue, J_dev, couplings)
    prg.packfV(queue, (nPairs*nB*nB,), (nB*nB,), J_dev, J_full_dev)
    prg.getEnergies(queue, (nsamples*NSEQ,), (WGSIZE,), 
                    J_full_dev, seqsamples_dev, energiessample_dev)
    countBimarg(bicount_dev, bimarg_dev, nsamples*NSEQ, seqsamples_dev)
    cl.enqueue_copy(queue, bimarg_model, bimarg_dev)
    cl.enqueue_copy(queue, bicount, bicount_dev)
    print "Unweighted Marginals: ", printsome(bimarg_model)
    save(os.path.join(outdir, 'preoptInitbimarg'), bimarg_model)
    save(os.path.join(outdir, 'preoptInitbicount'), bicount)

    rmsd = sqrt(mean((bimarg_target - bimarg_model)**2))
    ssd = sum((bimarg_target - bimarg_model)**2)
    print "RMSD: ", rmsd
    print "SSD: ", ssd

    #modify couplings a little
    gradientDescent(128)
    cl.enqueue_copy(queue, couplings, J_dev) 
    save(os.path.join(outdir, 'preoptPerturbedbimarg'), bimarg_model)
    save(os.path.join(outdir, 'preoptPerturbedJ'), couplings)
else:
    print "No Pre-optimization"

doFit(startseq, couplings)
print "Done!"

#Note that MCMC generation is split between nloop and nsteps.
#On some systems there is a watchdog timer that kills any kernel 
#that takes too long to finish, thus limiting the maximum nsteps. However,
#we avoid this by running the same kernel nloop times with smaller nsteps.
#If you set nsteps too high you will get a CL_OUT_OF_RESOURCES error.
#Restarting the MCMC kernel repeatedly also has the effect of recalculating
#the current energy from scratch, which re-zeros any floating point error
#that may build up during one kernel run.


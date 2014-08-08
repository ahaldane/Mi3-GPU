#!/usr/bin/env python2
from scipy import *
import scipy
from numpy.random import randint
import numpy.random
import pyopencl as cl
import sys, os, errno, glob, argparse
import load

numpy.random.seed(1234)

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


# Read in args and Data:
#=======================
parser = argparse.ArgumentParser(description='GD-MCMC')
parser.add_argument('bimarg')
parser.add_argument('gamma_d', type=float32)
parser.add_argument('gdsteps', type=uint32)
parser.add_argument('burnin', type=uint32)
parser.add_argument('nloop', type=uint32)
parser.add_argument('nsteps', type=uint32)
parser.add_argument('alpha')
parser.add_argument('startseq')
parser.add_argument('couplings', nargs='?')
parser.add_argument('-outdir')

args = parser.parse_args(sys.argv[1:])

if args.outdir:
    outdir = args.outdir
mkdir_p(outdir)


print "Initialization\n==============="

bimarg_target = loadtxt(args.bimarg, dtype='<f4')
L = int(((1+sqrt(1+8*bimarg_target.shape[0]))/2) + 0.5) 
nB = int(sqrt(bimarg_target.shape[1]) + 0.5) #+0.5 for rounding any fp error
nPairs = L*(L-1)/2;
n_couplings = nPairs*nB*nB
print "nBases {}  seqLen {}".format(nB, L)

                        # for example:
gamma_d = args.gamma_d  # 0.005
gdsteps = args.gdsteps  # 10
burnin = args.burnin    # 100
nloop = args.nloop      # 100
nsteps = args.nsteps    # 100

WGSIZE = 512
NGROUPS = 64
if WGSIZE < nB*nB:
    raise Exception("Error: WGSIZE must be larget than nB*nB")

print ("Running {} loops of {} sequences (x{}) with {} loops burnin per "
       "GD step, for {} GD steps.").format(
       nloop, nsteps, WGSIZE*NGROUPS, burnin, gdsteps)

bicount = empty((nPairs, nB*nB), dtype='<u4')
alpha = args.alpha
if len(alpha) != nB:
    print "Expected alphabet size {}, got {}".format(nB, len(alpha))
    exit()

ocouplings = zeros((nPairs, nB*nB), dtype='<f4')
ocouplings[bimarg_target == 0] = inf
if args.couplings:
    couplings = scipy.load(args.couplings)
    if couplings.dtype != dtype('<f4'):
        raise Exception("Couplings in wrong format")
else:
    print "Setting Initial couplings to 0"
    couplings = ocouplings.copy()

print "Initial Couplings: " + printsome(couplings) + "..."

pairI, pairJ = zip(*[(i,j) for i in range(L-1) for j in range(i+1,L)])
pairI, pairJ = array(pairI, dtype=uint32), array(pairJ, dtype=uint32)

if args.startseq == 'rand':
    startseq = randint(0, nB, size=L).astype('<u1')
else:
    startseq = array([alpha.index(c) for c in args.startseq], dtype='<u1')

print "Target Marginals: " + printsome(bimarg_target) + "..."

#Initialize GPU
#==============

print ""
platforms = cl.get_platforms()
for n,p in enumerate(platforms):
    print "Platform {} '{}'{}:".format(n, p.name, " (Chosen)" if n == 0 else "" )
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

ctx = cl.Context([devices[0]])
queue = cl.CommandQueue(ctx)

#sequences are padded to 32 bits
SWORDS = ((L-1)/4+1)  #number of words needed to store a sequence
SBYTES = (4*SWORDS)   #number of bytes needed to store a sequence

nseqs = WGSIZE*NGROUPS*nloop #number of seqs generated per GD
nseqload = 8192/SBYTES #number of seqs loaded into local memory in count kernel
#which should take up at most 8192 bytes (so in Geforce 580, 5 can fit in mem)
if nseqload > WGSIZE:
    nseqload = WGSIZE
while (NGROUPS*WGSIZE)%nseqload != 0:
    nseqload -= 1
#WG size for counting kernel (somewhat arbitrary)
WGSIZE_CT = 256
#number of work groups needed to process couplings
cplgrp = ((n_couplings-1)/WGSIZE_CT+1) 

srcfn = "metropolisS.cl"
with open(os.path.join(scriptPath, srcfn)) as f:
    src = f.read()
options = [('NGROUPS', NGROUPS), ('WGSIZE', WGSIZE), ('NSEQLOAD', nseqload), 
           ('SEED', randint(2**32)), ('nB', nB), ('L', L), ('nsteps', nsteps)]
optstr = " ".join(["-D {}={}".format(opt,val) for opt,val in options]) 
print "Options: ", optstr
print ""
extraopt = " -cl-nv-verbose -Werror -I {}".format(scriptPath)
prg = cl.Program(ctx, src).build(optstr + extraopt )
print "OpenCL Compilation Log:"
print prg.get_build_info(devices[0], cl.program_build_info.LOG)

def dumpPTX(prg):
    ptx = prg.get_info(cl.program_info.BINARIES)
    for n,p in enumerate(ptx):
        print "PTX: ", len(p)
        with open(os.path.join(outdir, 'ptx{}'.format(n)), 'wt') as f:
            f.write(p)
dumpPTX(prg)

#arrange seq memory for coalesced access
seqmem = zeros((SWORDS, WGSIZE*NGROUPS), dtype='<u4', order='C')

#converts seqs to uchars, padded to 32bits, assume GPU is little endian
bseqs = zeros((WGSIZE*NGROUPS, SBYTES), dtype='<u1', order='C')
def packSeqs(mem, seqs):
    bseqs[:,:L] = seqs  
    for i in range(SWORDS):
        mem[i,:] = bseqs.view(uint32)[:,i]
def unpackSeqs(mem):
    for i in range(SWORDS): #undo memory rearrangement
        bseqs.view(uint32)[:,i] = mem[i,:] 
    return bseqs[:,:L].copy()

#convert from format where every row is a unique ij pair (L choose 2 rows)
#to format with every pair, all orders (L^2 rows)
def packCouplings(couplings):
    fullcouplings = zeros((L*L,nB*nB), dtype='<f4', order='C')
    pairs = [(i,j) for i in range(L-1) for j in range(i+1,L)]
    for n,(i,j) in enumerate(pairs):
        fullcouplings[L*i + j,:] = couplings[n,:]
        fullcouplings[L*j + i,:] = couplings[n,:].reshape((nB,nB)).T.flatten()
    return fullcouplings

energies = zeros(WGSIZE*NGROUPS, dtype='<f4')

#allocate & init device memory
mf = cl.mem_flags
J_dev = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=packCouplings(couplings))
bicount_dev = cl.Buffer(ctx, mf.READ_WRITE, size=n_couplings*4)
bimarg_dev = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=bimarg_target)
run_seed_dev = cl.Buffer(ctx, mf.READ_WRITE, size=4)
seqmem_dev = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=seqmem)
energies_dev = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=energies)
pairI_dev = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=pairI)
pairJ_dev = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=pairJ)


#perform the computation!
#========================

print "\n\nMCMC Run\n========"

def writeStatus(n, rmsd, ssd, bicount, bimarg_model, couplings, seqs, startseq, energies):
    n = str(n)

    #print some details 
    disp = ["Start Seq: " + "".join([alpha[c] for c in startseq]),
            "RMSD: {}".format(rmsd),
            "SSD: {}".format(ssd),
            "Bicounts: " + printsome(bicount) + '...',
            "Marginals: " + printsome(bimarg_model) + '...',
            "Couplings: " + printsome(couplings) + "...",
            "Energies: Lowest =  {}, Mean = {}".format(min(energies), mean(energies))]
    dispstr = "\n".join(disp)
    
    mkdir_p(os.path.join(outdir, n))
    with open(os.path.join(outdir, n, 'info.txt'), 'wt') as f:
        print >>f, dispstr

    #save current state to file
    save(os.path.join(outdir, n, 'J'), couplings)
    savetxt(os.path.join(outdir, n, 'bicounts'), bicount, fmt='%d')
    load.writeSites(os.path.join(outdir, n, 'finalseqs'), 
                    seqs, alpha, param={'alpha': alpha})

    print dispstr

def getEnergies(s, couplings):
    pairenergy = zeros(s.shape[0], dtype='<f4')
    for n,(i,j) in enumerate([(i,j) for i in range(L-1) for j in range(i+1,L)]):
        pairenergy += couplings[n,nB*s[:,i] + s[:,j]]
    return pairenergy

def doFit(startseq, couplings):
    nproj = 8
    kernel_seed = array([0], dtype=int32)

    lastrmsd = inf
    for i in range(gdsteps):
        #equilibrate until we are back at original rmsd 3 times
        nMinRMSD = 0
        equilN = 0
        while equilN < nproj or nMinRMSD < nproj:
            rmsd, (seqs, energies) = singleStep('{}_{}equil'.format(i,equilN), kernel_seed, couplings, startseq)
            startseq = seqs[argmin(energies)]
            equilN += 1
            if rmsd < lastrmsd:
                nMinRMSD += 1
        
        #run nprogs steps, recording couplings
        recJ = []
        for l in range(nproj):
            rmsd, (seqs, energies) = singleStep('{}_{}sample'.format(i,l), kernel_seed, couplings, startseq)
            startseq = seqs[argmin(energies)]
            recJ.append(couplings.copy())
        
        #project couplings some number of steps ahead
        projectionFactor = 8

        recJ = vstack([j.flatten() for j in recJ])
        polys = polyfit(arange(nproj), recJ, 1)
        projJ = array([polyval(polys[:,l], nproj + projectionFactor*nproj) for l in range(polys.shape[1])])
        couplings[:,:] = projJ.reshape(couplings.shape)

        print ""
        print "Projecting forward {} steps.".format(projectionFactor*nproj)
        print "Projected Couplings: " + printsome(couplings) + "..."
        save(os.path.join(outdir, 'Jproj_{}'.format(i)), couplings)
        
        lastrmsd = rmsd

        cl.enqueue_copy(queue, J_dev, couplings)

def runMCMC(kernel_seed):
    cl.enqueue_copy(queue, run_seed_dev, kernel_seed)
    kernel_seed[0] += 1
    prg.metropolis(queue, (WGSIZE*NGROUPS,), (WGSIZE,), 
                          J_dev, run_seed_dev, seqmem_dev, energies_dev)

def singleStep(runName, kernel_seed, couplings, startseq):
    global bicount 
    print ""
    print "Gradient Descent step {}".format(runName)
    #print "Starting from E={}".format(getEnergies(startseq[newaxis,:], couplings)[0])
    
    #randomize initial sequences
    packSeqs(seqmem, tile(startseq, (NGROUPS*WGSIZE,1)))
    #packSeqs(seqmem, randint(0, nB, size=(WGSIZE*NGROUPS, L)))
    cl.enqueue_copy(queue, seqmem_dev, seqmem)

    #load in latest couplings
    cl.enqueue_copy(queue, J_dev, packCouplings(couplings))

    #burnin loops
    for j in range(burnin):
        runMCMC(kernel_seed)
    
    #counting MCMC loop
    bicount.fill(0)
    cl.enqueue_copy(queue, bicount_dev, bicount)
    for j in range(nloop):
        runMCMC(kernel_seed)
        prg.countSeqs(queue, (WGSIZE_CT*cplgrp,), (WGSIZE_CT,), 
                      bicount_dev, seqmem_dev, pairI_dev, pairJ_dev)
    
    #read out data
    cl.enqueue_copy(queue, bicount, bicount_dev)
    bicount += 1
    bimarg_model = bicount/float32(sum(bicount[0]))
    rmsd = sqrt(mean((bimarg_target - bimarg_model)**2))
    ssd = sum((bimarg_target - bimarg_model)**2)
    #read out sequence info
    cl.enqueue_copy(queue, seqmem, seqmem_dev)
    seqs = unpackSeqs(seqmem)
    cl.enqueue_copy(queue, energies, energies_dev)
    energies2 = getEnergies(seqs, couplings)
    #print "CL: ", energies[:5]
    #print "Numpy: ", energies2[:5]
    
    #update couplings
    couplings += -gamma_d*(bimarg_target - bimarg_model)/bimarg_model
    couplings = couplings - mean(couplings, axis=1)[:,newaxis]

    writeStatus(runName, rmsd, ssd, bicount, bimarg_model, couplings, seqs, startseq, energies)
    return rmsd, (seqs, energies)

doFit(startseq, couplings)

#kernel_seed = array([0], dtype=int32)
##startseq = randint(0, nB, size=L).astype('<u1')
#startseq = zeros(L).astype('<u1')
#singleStep('test', kernel_seed, couplings, startseq)

#Note that loop control is split between nloop and nsteps. This is because
#on some (many) systems there is a watchdog timer that kills any kernel 
#that takes too long to finish, thus limiting the maximum nsteps. However,
#we avoid this by running the same kernel nloop times with smaller nsteps.
#If you set nsteps too high you will get a CL_OUT_OF_RESOURCES error.

print "Done!"


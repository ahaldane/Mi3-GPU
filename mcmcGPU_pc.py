#!/usr/bin/env python2
from scipy import *
import scipy
from numpy.random import randint
import pyopencl as cl
import sys, os, errno, glob, argparse
import load

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

WGSIZE, NGROUPS = 256, 32

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
parser.add_argument('couplings', nargs='?')

args = parser.parse_args(sys.argv[1:])


print "Initialization\n==============="

bimarg = loadtxt(args.bimarg, dtype='<f4')
L = int(((1+sqrt(1+8*bimarg.shape[0]))/2) + 0.5) 
nB = int(sqrt(bimarg.shape[1]) + 0.5) #+0.5 for rounding any fp error
nPairs = L*(L-1)/2;
n_couplings = nPairs*nB*nB
print "nBases {}  seqLen {}".format(nB, L)

                        # for example:
gamma_d = args.gamma_d  # 0.005
gdsteps = args.gdsteps  # 10
burnin = args.burnin    # 100
nloop = args.nloop      # 100
nsteps = args.nsteps    # 128

print ("Running {} loops of {} sequences (x{}) with {} loops burnin per "
       "GD step, for {} GD steps.").format(
       nloop, nsteps, WGSIZE*NGROUPS, burnin, gdsteps)

bicount = empty((nPairs, nB*nB), dtype='<u4')
alpha = args.alpha
if len(alpha) != nB:
    print "Expected alphabet size {}, got {}".format(nB, len(alpha))
    exit()

ocouplings = zeros((nPairs, nB*nB), dtype='<f4')
ocouplings[bimarg == 0] = inf
if args.couplings:
    couplings = scipy.load(args.couplings)
    if couplings.dtype != dtype('<f4'):
        raise Exception("Couplings in wrong format")
else:
    print "Setting Initial couplings to 0"
    couplings = ocouplings

print "Initial Couplings: " + printsome(couplings) + "..."

pairI, pairJ = zip(*[(i,j) for i in range(L-1) for j in range(i+1,L)])
pairI, pairJ = array(pairI, dtype=uint32), array(pairJ, dtype=uint32)

mkdir_p(outdir)

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
#number of work groups needed to process couplings
cplgrp = ((n_couplings-1)/WGSIZE+1) 

if nseqload == WGSIZE:
    print 'Using "Local" OpenCL code for short sequences'
    srcfn = "metropolisLocal.cl"
else:
    print 'Using "Global" OpenCL code for long sequences'
    srcfn = "metropolisGlobal.cl"

with open(os.path.join(scriptPath, srcfn)) as f:
    src = f.read()
options = [('NGROUPS', NGROUPS), ('WGSIZE', WGSIZE), ('NSEQLOAD', nseqload), 
           ('SEED', randint(2**32)), ('nB', nB), ('L', L), ('nsteps', nsteps), 
           ('nseqs', nseqs), ('gamma', gamma_d)]
optstr = " ".join(["-D {}={}".format(opt,val) for opt,val in options]) 
print "Options: ", optstr
print ""
extraopt = " -cl-nv-verbose -Werror -I {}".format(scriptPath)
prg = cl.Program(ctx, src).build(optstr + extraopt )
print "OpenCL Compilation Log:"
print prg.get_build_info(devices[0], cl.program_build_info.LOG)

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
    return bseqs[:,:L]

#allocate & init device memory
mf = cl.mem_flags
J_dev = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=couplings)
bicount_dev = cl.Buffer(ctx, mf.READ_WRITE, size=n_couplings*4)
bimarg_dev = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=bimarg)
run_seed_dev = cl.Buffer(ctx, mf.READ_WRITE, size=4)
seqmem_dev = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=seqmem)
pairI_dev = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=pairI)
pairJ_dev = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=pairJ)


#perform the computation!
#========================

print "\n\nMCMC Run\n========"

def writeStatus(n, rmsd, ssd, bicount, couplings, seqs):
    n = str(n)

    #print some details 
    disp = ["RMSD: {}".format(rmsd),
            "SSD: {}".format(ssd),
            "Bicounts: " + printsome(bicount) + '...',
            "Marginals: " + printsome(bicount/float(nseqs)) + '...',
            "Couplings: " + printsome(couplings) + "..."]
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

nonzeroJ = isfinite(couplings.flatten())
def doFit():
    nproj = 8
    kernel_seed = array([0], dtype=int32)
    global couplings
    lastrmsd = inf
    for i in range(gdsteps):
        #equilibrate until we are back at original rmsd 3 times
        nMinRMSD = 0
        equilN = 0
        while equilN < nproj or nMinRMSD < 3:
            rmsd = singleStep('{}_{}equil'.format(i,equilN), kernel_seed)
            equilN += 1
            if rmsd < lastrmsd:
                nMinRMSD += 1
        
        #run nprogs steps, recording couplings
        recJ = []
        for l in range(nproj):
            rmsd = singleStep('{}_{}sample'.format(i,l), kernel_seed)
            recJ.append(couplings.copy())
        
        #project couplings some number of steps ahead (XXX clean this up please)
        projectionFactor = 8
        recJ = vstack([j.flatten() for j in recJ])
        polys = polyfit(arange(nproj), recJ[:,nonzeroJ], 1)
        projJ = array([polyval(polys[:,l], nproj + projectionFactor*nproj) for l in range(polys.shape[1])])
        couplings.reshape(couplings.size)[nonzeroJ] = projJ
        print ""
        print "Projecting forward {} steps.".format(projectionFactor*nproj)
        print "Projected Couplings: " + printsome(couplings) + "..."
        save(os.path.join(outdir, 'Jproj_{}'.format(i)), couplings)
        
        lastrmsd = rmsd

        cl.enqueue_copy(queue, J_dev, couplings)

def singleStep(runName, kernel_seed):
    print ""
    print "Gradient Descent step {}".format(runName)
    
    #randomize initial sequences
    #(WARNING: this does not work if any couplings are inf!!!)
    #(In that case use mcmcSGPU.py)
    packSeqs(seqmem, randint(0, nB, size=(WGSIZE*NGROUPS, L)))
    cl.enqueue_copy(queue, seqmem_dev, seqmem)

    #burnin loops
    for j in range(burnin):
        cl.enqueue_copy(queue, run_seed_dev, kernel_seed)
        kernel_seed[0] += 1
        prg.metropolis(queue, (WGSIZE*NGROUPS,), (WGSIZE,), 
                              J_dev, run_seed_dev, seqmem_dev)
    
    #counting MCMC loop
    prg.zeroBicounts(queue, (WGSIZE*cplgrp,), (WGSIZE,), bicount_dev)
    for j in range(nloop):
        cl.enqueue_copy(queue, run_seed_dev, kernel_seed )
        kernel_seed[0] += 1
        prg.metropolis(queue, (WGSIZE*NGROUPS,), (WGSIZE,), 
                       J_dev, run_seed_dev, seqmem_dev)
        prg.countSeqs(queue, (WGSIZE*cplgrp,), (WGSIZE,), 
                      bicount_dev, seqmem_dev, pairI_dev, pairJ_dev)

    cl.enqueue_copy(queue, bicount, bicount_dev)
    rmsd = sqrt(mean((bimarg - bicount/float32(nseqs))**2))
    ssd = sum((bimarg - bicount/float32(nseqs))**2)

    prg.updateCouplings(queue, (WGSIZE*cplgrp,), (WGSIZE,), 
                        bimarg_dev, bicount_dev, J_dev)

    cl.enqueue_copy(queue, couplings, J_dev)
    cl.enqueue_copy(queue, seqmem, seqmem_dev)
    writeStatus(runName, rmsd, ssd, bicount, couplings, unpackSeqs(seqmem))
    return rmsd

doFit()

#Note that loop control is split between nloop and nsteps. This is because
#on some (many) systems there is a watchdog timer that kills any kernel 
#that takes too long to finish, thus limiting the maximum nsteps. However,
#we avoid this by running the same kernel nloop times with smaller nsteps.
#If you set nsteps too high you will get a CL_OUT_OF_RESOURCES error.

print "Done!"


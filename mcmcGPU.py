#!/usr/bin/env python2
from scipy import *
from numpy.random import randint
import pyopencl as cl
import sys, os
import load

os.environ['PYOPENCL_COMPILER_OUTPUT'] = '0'
os.environ['PYOPENCL_NO_CACHE'] = '1'
ndisp = 5

WGSIZE, NGROUPS = 256, 32

# Read in args and Data:
print "Initialization\n==============="

args = iter(sys.argv)
args.next()

bimarg = loadtxt(args.next(), dtype='<f4')
L = int(((1+sqrt(1+8*bimarg.shape[0]))/2) + 0.5) 
nB = int(sqrt(bimarg.shape[1]) + 0.5) #+0.5 for rounding any fp error
nPairs = L*(L-1)/2;
n_couplings = nPairs*nB*nB
print "nBases {}  seqLen {}".format(nB, L)

                                # for example:
gamma_d = float32(args.next())  # 0.005
gdsteps = uint32(args.next())   # 10
burnstart = uint32(args.next()) # 100
burnin = uint32(args.next())    # 100
nloop = uint32(args.next())     # 100
nsteps = uint32(args.next())    # 100

print ("Running {} loops of {} sequences (x{}) with {} loops burnin per "
       "GD step, for {} GD steps. Running {} loops of pre-GD burnin.").format(
       nloop, nsteps, WGSIZE*NGROUPS, burnin, gdsteps, burnstart)
print ("Re-initializing to random seqs each GD step by doing 100 loops of "
       "neutral sampling.")

bicount = empty((nPairs, nB*nB), dtype='<u4')
oseqs, seqinfo = load.loadSites(args.next())
params = seqinfo[2]
alpha = params['alpha']

if oseqs.shape[1] != L:
    print "Expected sequence length {}, got {}".format(L, oseqs.shape[1])
    exit()
if oseqs.shape[0] != WGSIZE*NGROUPS:
    print "Expected {} sequences, got {}".format(WGSIZE*NGROUPS, oseqs.shape[0])
    exit()
if len(alpha) != nB:
    print "Expected alphabet size {}, got {}".format(nB, len(alpha))
    exit()

ocouplings = zeros((nPairs, nB*nB), dtype='<f4')
ocouplings[bimarg == 0] = inf
try:
    couplings = load(args.next())
    if couplings.dtype != dtype('<f4'):
        raise Exception("Couplings in wrong format")
except:
    couplings = ocouplings
print "Initial Couplings: " + " ".join(map(str,couplings[0,:ndisp])) + "..."

pairI, pairJ = zip(*[(i,j) for i in range(L-1) for j in range(i+1,L)])
pairI, pairJ = array(pairI, dtype=uint32), array(pairJ, dtype=uint32)

#Initialize GPU

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

#sequenes are padded to 32 bits
SWORDS = ((L-1)/4+1)  #number of words needed to store a sequence
SBYTES = (4*SWORDS)   #number of bytes needed to store a sequence

nseqs = WGSIZE*NGROUPS*nloop #number of seqs generated per GD
nseqload = 8192/SBYTES #number of seqs loaded into local memory in count kernel
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

with open(srcfn) as f:
    src = f.read()
options = [('NGROUPS', NGROUPS), ('WGSIZE', WGSIZE), ('NSEQLOAD', nseqload), 
           ('SEED', randint(2**32)), ('nB', nB), ('L', L), ('nsteps', nsteps), 
           ('nseqs', nseqs), ('gamma', gamma_d)]
optstr = " ".join(["-D {}={}".format(opt,val) for opt,val in options]) 
print "Options: ", optstr
print ""
prg = cl.Program(ctx, src).build(optstr + " -cl-nv-verbose -Werror")
print "OpenCL Compilation Log:"
print prg.get_build_info(devices[0], cl.program_build_info.LOG)

#convert seqs to uchars, padded to 32bits, assume GPU is little endian
bseqs = zeros((oseqs.shape[0], SBYTES), dtype='<u1', order='C')
bseqs[:,:L] = oseqs  
#arrange seq memory for coalesced access
seqmem = zeros((SWORDS, WGSIZE*NGROUPS), dtype='<u4', order='C')
for i in range(SWORDS):
    seqmem[i,:] = bseqs.view(uint32)[:,i]
ccseqmem = seqmem.copy()

mf = cl.mem_flags
J_dev = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=couplings)
bicount_dev = cl.Buffer(ctx, mf.READ_WRITE, size=n_couplings*4)
bimarg_dev = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=bimarg)
run_seed_dev = cl.Buffer(ctx, mf.READ_WRITE, size=4)
seqmem_dev = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=seqmem)
pairI_dev = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=pairI)
pairJ_dev = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=pairJ)


#perform the computation!
print "\n\nMCMC Run\n========"

def writeStatus():
    cl.enqueue_copy(queue, couplings, J_dev)
    cl.enqueue_copy(queue, bicount, bicount_dev)
    cl.enqueue_copy(queue, seqmem, seqmem_dev)
    
    #print out rmsd
    rmsd = sqrt(mean((bimarg - bicount/float32(nseqs))**2))
    print "RMSD: {}".format(rmsd)
    rmsd = sum((bimarg - bicount/float32(nseqs))**2)
    print "SSD: {}".format(rmsd)
    
    #print some details to stdout
    print "Bicounts: " + " ".join(map(str,bicount[0,:ndisp])) + '...'
    print "Marginals: " + " ".join(map(str,bicount[0,:ndisp]/float(nseqs))) + '...'
    print "Couplings: " + " ".join(map(str,couplings[0,:ndisp])) + "..."

    #save current state to file
    savetxt('result', couplings)
    save('binresult', couplings)
    savetxt('bicounts', bicount, fmt='%d')
    for i in range(SWORDS): #undo memory rearrangement
        bseqs.view(uint32)[:,i] = seqmem[i,:] 
    load.writeSites('finalsequences', bseqs[:,:L], alpha, param=params)

#initial burnin
kernel_seed = array([0], dtype=int32)
print "Initial Burnin for {} steps:".format(burnstart)
for j in range(burnstart):
    cl.enqueue_copy(queue, run_seed_dev, kernel_seed )
    kernel_seed[0] += 1
    prg.metropolis(queue,( WGSIZE*NGROUPS,),( WGSIZE,), J_dev, run_seed_dev, seqmem_dev)

#main loop
for i in range(gdsteps):
    print ""
    print "Gradient Descent step {}".format(i)

    cl.enqueue_copy(queue, seqmem_dev, ccseqmem)
    cl.enqueue_copy(queue, couplings, J_dev)
    cl.enqueue_copy(queue, J_dev, ocouplings)
    
    #neutral initialization loops
    for j in range(100):
        cl.enqueue_copy(queue, run_seed_dev, kernel_seed)
        kernel_seed[0] += 1
        prg.metropolis(queue,( WGSIZE*NGROUPS,),( WGSIZE,), J_dev, run_seed_dev, seqmem_dev)

    cl.enqueue_copy(queue, J_dev, couplings)

    #burnin loops
    for j in range(burnin):
        cl.enqueue_copy(queue, run_seed_dev, kernel_seed)
        kernel_seed[0] += 1
        prg.metropolis(queue,( WGSIZE*NGROUPS,),( WGSIZE,), J_dev, run_seed_dev, seqmem_dev)
    
    #counting MCMC loop
    prg.zeroBicounts(queue,( WGSIZE*cplgrp,),( WGSIZE,), bicount_dev)
    for j in range(nloop):
        cl.enqueue_copy(queue, run_seed_dev, kernel_seed )
        kernel_seed[0] += 1
        prg.metropolis(queue,( WGSIZE*NGROUPS,),( WGSIZE,), J_dev, run_seed_dev, seqmem_dev)
        prg.countSeqs(queue,( WGSIZE*cplgrp,),( WGSIZE,), bicount_dev, seqmem_dev, pairI_dev, pairJ_dev)
    prg.updateCouplings(queue,( WGSIZE*cplgrp,),( WGSIZE,), bimarg_dev, bicount_dev, J_dev)

    writeStatus()

#Note that loop control is split between nloop and nsteps. This is because
#on some (many) systems there is a watchdog timer that kills any kernel 
#that takes too long to finish, thus limiting the maximum nsteps. However,
#we avoid this by running the same kernel nloop times with smaller nsteps.
#If you set nsteps too high you will get a CL_OUT_OF_RESOURCES error.

print "Done!"


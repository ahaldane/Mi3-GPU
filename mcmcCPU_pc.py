#!/usr/bin/env python2
from scipy import *
import scipy
from numpy.random import randint
import sys, os, errno, glob, argparse
import load
import subprocess

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc: 
        if not (exc.errno == errno.EEXIST and os.path.isdir(path)):
            raise
scriptPath = os.path.dirname(os.path.realpath(__file__))
outdir = 'output'
printsome = lambda a: " ".join(map(str,a.flatten()[:5]))

# Read in args and Data:
#=======================
parser = argparse.ArgumentParser(description='GD-MCMC')
parser.add_argument('bimarg')
parser.add_argument('gamma_d', type=float32)
parser.add_argument('gdsteps', type=uint32)
parser.add_argument('burnin', type=uint32)
parser.add_argument('nloop', type=uint32)
parser.add_argument('nsteps', type=uint32)
parser.add_argument('nprocs', type=uint32)
parser.add_argument('alpha')
parser.add_argument('couplings', nargs='?')

args = parser.parse_args(sys.argv[1:])


print "Initialization\n==============="

bimarg = loadtxt(args.bimarg, dtype='<f8')
L = int(((1+sqrt(1+8*bimarg.shape[0]))/2) + 0.5) 
nB = int(sqrt(bimarg.shape[1]) + 0.5) #+0.5 for rounding any fp error
nPairs = L*(L-1)/2;
nCouplings = nPairs*nB*nB
print "nBases {}  seqLen {}".format(nB, L)

                        # for example:
gamma_d = args.gamma_d  # 0.005
gdsteps = args.gdsteps  # 10
burnin = args.burnin    # 100
nloop = args.nloop      # 100
nsteps = args.nsteps    # 100
nprocs = args.nprocs

nseqs = nloop*nprocs

print ("wGenerating {} seqs with {} burnin seqs per GD step, skipping every "
       "{} sequences, on {} processors, for {} GD steps.").format(
       nloop, burnin, nsteps, nprocs, gdsteps)

bicount = empty((nPairs, nB*nB), dtype='<u4')
alpha = args.alpha
if len(alpha) != nB:
    print "Expected alphabet size {}, got {}".format(nB, len(alpha))
    exit()

ocouplings = zeros((nPairs, nB*nB), dtype='<f8')
ocouplings[bimarg == 0] = inf
if args.couplings:
    couplings = scipy.load(args.couplings).astype('<f8')
    if couplings.dtype != dtype('<f8'):
        raise Exception("Couplings in wrong format")
else:
    print "Setting Initial couplings to 0"
    couplings = ocouplings

print "Initial Couplings: " + printsome(couplings) + "..."

pairI, pairJ = zip(*[(i,j) for i in range(L-1) for j in range(i+1,L)])
pairI, pairJ = array(pairI, dtype=uint32), array(pairJ, dtype=uint32)

mkdir_p(outdir)

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

def runMCMC(couplings, startseqs, nsteps, nloops, kernel_seed):
    kernel_seed[0] += 1

    processes = []
    for n,seq in enumerate(startseqs):
        outfile = os.tmpfile()
        infile = os.tmpfile()
        seq.astype('u1').tofile(infile)
        couplings.tofile(infile)
        infile.seek(0)
        #print " ".join(['./mcmcCPUbin'] + [str(x) for x in [L, nB, nloops, nsteps, kernel_seed[0], n]])
        p = subprocess.Popen(['./mcmcCPUbin'] + [str(x) for x in [L, nB, nloops, nsteps, kernel_seed[0], n]], stdout=outfile, stdin=infile)
        infile.close()
        processes.append((p, outfile))
    
    bicounts = zeros(couplings.shape, dtype=uint)
    seqs = []
    for p, f in processes:
        p.wait()
        f.seek(0)
        seqs.append(fromfile(f, dtype='u1', count=L))
        bicounts += fromfile(f, dtype='<u8', count=nCouplings).reshape(couplings.shape)
        f.close()
        #print "Done"
    
    return array(seqs), bicounts

def singleStep(runName, kernel_seed):
    print ""
    print "Gradient Descent step {}".format(runName)
    
    #randomize initial sequences
    #(WARNING: this does not work if any couplings are inf!!!)
    seqs = randint(0, nB, size=(nprocs,L))
    seqs, bicounts = runMCMC(couplings, seqs, burnin*nsteps, 1, kernel_seed)
    seqs, bicounts = runMCMC(couplings, seqs, nsteps, nloop, kernel_seed)
    
    savetxt('bicounts', bicounts)
    bi_obs = bicounts/float(nseqs)

    rmsd = sqrt(mean((bimarg - bi_obs)**2))
    ssd = sum((bimarg - bi_obs)**2)

    writeStatus(runName, rmsd, ssd, bicounts, couplings, seqs)
    return rmsd

doFit()

#Note that loop control is split between nloop and nsteps. This is because
#on some (many) systems there is a watchdog timer that kills any kernel 
#that takes too long to finish, thus limiting the maximum nsteps. However,
#we avoid this by running the same kernel nloop times with smaller nsteps.
#If you set nsteps too high you will get a CL_OUT_OF_RESOURCES error.

print "Done!"


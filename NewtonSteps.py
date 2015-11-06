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
from mcmcGPU import readGPUbufs

################################################################################
# Set up enviroment and some helper functions

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc: 
        if not (exc.errno == errno.EEXIST and os.path.isdir(path)):
            raise

printsome = lambda a: " ".join(map(str,a.flatten()[-5:]))

################################################################################
#Helper funcs

def writeStatus(name, rmsd, ssr, wdf, bicount, bimarg_model, couplings, 
                seqs, startseq, energies, alpha, outdir, log):

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
        f.write(dispstr)

    #save current state to file
    savetxt(os.path.join(outdir, name, 'bicounts'), bicount, fmt='%d')
    save(os.path.join(outdir, name, 'bimarg'), bimarg_model)
    save(os.path.join(outdir, name, 'energies'), energies)
    for n,seqbuf in enumerate(seqs):
        seqload.writeSeqs(os.path.join(outdir, name, 'seqs-{}'.format(n)), 
                          seqbuf, alpha)

    log(dispstr)

def sumarr(arrlist):
    #low memory usage (rather than sum(arrlist, axis=0))
    tot = arrlist[0].copy()
    for a in arrlist[1:]:
        np.add(tot, a, tot)
    return tot

def meanarr(arrlist):
    return sumarr(arrlist)/len(arrlist)

################################################################################
#local optimization related code

def newtonStep(n, bimarg_target, gamma, pc, jclamp, gpus, log):
    # expects the back buffers to contain current couplings & bimarg,
    # will overwrite front buffers

    # calculate perturbed marginals
    for gpu in gpus:
        # note: updateJPerturb should give same result on all GPUs
        # overwrites J front using bi back and J back
        gpu.updateJPerturb(gamma, pc, jclamp) 

    for gpu in gpus:
        gpu.swapBuf('J') #temporarily put trial J in back buffer
        gpu.perturbMarg() #overwrites bi front using J back
        gpu.swapBuf('J')
    # at this point, front = trial param, back = last accepted param
    
    #read out result and update bimarg
    res = readGPUbufs(['bi front', 'neff', 'weights'], gpus)
    bimargb, Neffs, weightb = res
    Neff = sum(Neffs)
    bimarg_model = sumarr([N*buf for N,buf in zip(Neffs, bimargb)])/Neff
    weights = concatenate(weightb)
    SSR = sum((bimarg_model.flatten() - bimarg_target.flatten())**2)
    trialJ = gpus[0].getBuf('J front').read()
    
    #display result
    log("")
    log(("{}  ssr: {}  Neff: {:.1f} wspan: {:.3g}:{:.3g}").format(
         n, SSR, Neff, min(weights), max(weights)))
    log("    trialJ:", printsome(trialJ))
    log("    bimarg:", printsome(bimarg_model))
    log("   weights:", printsome(weights))

    if isinf(Neff) or Neff == 0:
        raise Exception("Error: Divergence. Decrease gamma or increase "
                        "pc-damping")
    
    # dump profiling info if profiling is turned on
    for gpu in gpus:
        gpu.logProfile()

    return SSR, bimarg_model

def iterNewton(param, gpus, log):
    gamma = gamma0 = param.gamma0
    newtonSteps = param.newtonSteps
    pc, jclamp = param.pcdamping, param.jclamp
    gammasteps = 16
    bimarg_target = param.bimarg

    # setup front and back buffers. Back buffers should contain last accepted
    # values, front buffers vonctain trial values.
    for gpu in gpus:
        gpu.calcEnergies('large', 'main')
        gpu.copyBuf('J main', 'J back')
        gpu.copyBuf('J main', 'J front')
        gpu.copyBuf('bi main', 'bi back')

    log("Local target: ", printsome(bimarg_target))
    log("Local optimization:")
    
    # newton updates
    n = 1
    lastSSR = inf
    nrejects = 0
    for i in range(newtonSteps): 

        # increase gamma every gammasteps steps
        if i != 0 and i % gammasteps == 0:
            if nrejects == gammasteps:
                log("Too many ssr increases. Stopping newton updates.")
                break
            gamma = gamma*2
            log("Increasing gamma to {}".format(gamma))
            nrejects = 0
        
        # do newton step
        ssr, bimarg_model = newtonStep(n, bimarg_target, gamma, pc, jclamp, 
                                       gpus, log)

        # accept move if ssr decreases, reject otherwise
        if ssr <= lastSSR:  # accept move
            #keep this step, and store current J and bm to back buffer
            for gpu in gpus:
                gpu.storeBuf('J') #copy trial J to back buffer
                gpu.setBuf('bi front', bimarg_model)
                gpu.storeBuf('bi')
            n += 1
            lastSSR = ssr
        else: # reject move
            gamma = gamma/2
            log("Reducing gamma to {} and repeating step".format(gamma))
            nrejects += 1
            if gamma < gamma0/64:
                log("gamma decreased too much relative to gamma0. Stopping")
                break

    # return back buffer, which contains last accepted move
    return (gpus[0].getBuf('J back').read(), 
            gpus[0].getBuf('bi back').read())
    
################################################################################

#pre-optimization steps
def preOpt(param, gpus, log):
    couplings = param.couplings
    outdir = param.outdir
    alpha = param.alpha
    bimarg_target = param.bimarg

    log("Processing sequences...")
    for gpu in gpus:
        gpu.setBuf('J main', couplings)
        gpu.calcEnergies('large', 'main')
        gpu.calcBimarg('large')
    res = readGPUbufs(['bi main', 'bicount', 'seq large'], gpus)
    bimarg, bicount, seqs = meanarr(res[0]), sumarr(res[1]), res[2]
    print(concatenate(readGPUbufs(['E large'], gpus)[0]))
    
    #store initial setup
    mkdir_p(os.path.join(outdir, 'preopt'))
    log("Unweighted Marginals: ", printsome(bimarg))
    save(os.path.join(outdir, 'preopt', 'initbimarg'), bimarg)
    save(os.path.join(outdir, 'preopt', 'initBicont'), bicount)
    for n,s in enumerate(seqs):
        seqload.writeSeqs(os.path.join(outdir, 'preopt', 'seqs-'+str(n)), 
                          s, alpha)

    rmsd = sqrt(mean((bimarg_target - bimarg)**2))
    ssr = sum((bimarg_target - bimarg)**2)
    wdf = sum(bimarg_target*abs(bimarg_target - bimarg))
    log("S RMSD: {: 9.7f}  SSR: {: 9.5f}  wDf: {: 9.5f}".format(rmsd,ssr,wdf))

    #modify couplings a little
    couplings, bimarg_p = localDescent(perturbSteps, gamma0, gpus)
    save(os.path.join(outdir, 'preopt', 'perturbedbimarg'), bimarg_p)
    save(os.path.join(outdir, 'preopt', 'perturbedJ'), couplings)


def MCMCstep(runName, startseq, couplings, param, gpus, log):
    nloop = param.equiltime
    nsamples = param.nsamples
    nsampleloops = param.sampletime
    trackequil = param.trackequil
    outdir = param.outdir
    alpha, L, nB = param.alpha, param.L, param.nB
    bimarg_target = param.bimarg
    outdir = param.outdir

    log("")
    log("Gradient Descent step {}".format(runName))

    #re-distribute energy among couplings
    #(not really needed, but makes nicer output and might prevent
    # numerical inaccuracy, but also shifts all seq energies)
    log("(Re-centering gauge of couplings)")
    couplings = fieldlessGaugeEven(zeros((L,nB)), couplings)[1]

    mkdir_p(os.path.join(outdir, runName))
    save(os.path.join(outdir, runName, 'J'), couplings)
    with open(os.path.join(outdir, runName, 'startseq'), 'wt') as f:
        f.write("".join(alpha[c] for c in startseq))

    #get ready for MCMC
    for gpu in gpus:
        gpu.resetSeqs(startseq)
        gpu.setBuf('J main', couplings)
    
    #equilibration MCMC
    if trackequil == 0:
        #keep nloop iterator on outside to avoid filling queue with only 1 gpu
        for i in range(nloop):
            for gpu in gpus:
                gpu.runMCMC()
    else:
        #note: sync necessary with trackequil (may slightly affect performance)
        mkdir_p(os.path.join(outdir, runName, 'equilibration'))
        for j in range(nloop/trackequil):
            for i in range(trackequil):
                for gpu in gpus:
                    gpu.runMCMC()
            for gpu in gpus:
                gpu.calcBimarg('small')
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
            gpu.storeSeqs(offset=j*gpu.nseq['small'])
    
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
                couplings, sampledseqs, startseq, sampledenergies, 
                alpha, outdir, log)
    print(sampledenergies)
    
    #compute new J using local newton updates (in-place on GPU)
    couplings, bimarg_p = iterNewton(param, gpus, log)
    save(os.path.join(outdir, runName, 'predictedBimarg'), bimarg_p)

    #choose seed sequence for next round
    rseq_ind = numpy.random.randint(0, len(sampledenergies))
    nseq = gpus[0].nseq['large'] #assumes all gpus the same
    rseq = sampledseqs[rseq_ind/nseq][rseq_ind%nseq]

    return rseq, couplings

def newtonMCMC(param, gpus, log):
    startseq = param.startseq
    couplings = param.couplings

    if startseq is None:
        raise Exception("Error: Potts inference requires a starting sequence")
    
    # pre-optimization
    if param.preopt:
        if param.seqs is None:
            raise Exception("Error: sequence buffers must be filled for "
                            "pre-optimization") 
                            #this doesn't actually check that....
        preOpt(param, gpus, log)
    else:
        log("No Pre-optimization")
    
    # copy target bimarg to gpus
    for gpu in gpus:
        gpu.setBuf('bi target', param.bimarg)
    
    # solve using newton-MCMC
    for i in range(param.mcmcsteps):
        runname = 'run_{}'.format(i)
        startseq, couplings = MCMCstep(runname, startseq, couplings, 
                                       param, gpus, log)

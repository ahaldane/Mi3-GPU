#!/usr/bin/env python2
from __future__ import print_function
from scipy import *
import scipy
import scipy.stats.distributions as dists
import numpy as np
from numpy.random import randint, shuffle
import numpy.random
import pyopencl as cl
import pyopencl.array as cl_array
import sys, os, errno, glob, argparse, time
import ConfigParser
import seqload
from scipy.optimize import leastsq
from changeGauge import zeroGauge, zeroJGauge, fieldlessGaugeEven
from mcmcGPU import readGPUbufs

def unimarg(bimarg):
    L, nB = seqsize_from_param_shape(bimarg.shape)
    ff = bimarg.reshape((L*(L-1)/2,nB,nB))
    marg = (array([sum(ff[0],axis=1)] + 
            [sum(ff[n],axis=0) for n in range(L-1)]))
    return marg/(sum(marg,axis=1)[:,newaxis]) # correct any fp errors

def indep_bimarg(bimarg):
    f = unimarg(bimarg)
    L = f.shape[0]
    return array([outer(f[i], f[j]).flatten() for i in range(L-1) 
                                    for j in range(i+1,L)])

def seqsize_from_param_shape(shape):
    L = int(((1+sqrt(1+8*shape[0]))/2) + 0.5) 
    nB = int(sqrt(shape[1]) + 0.5) 
    return L, nB

################################################################################
# Set up enviroment and some helper functions

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc: 
        if not (exc.errno == errno.EEXIST and os.path.isdir(path)):
            raise

class attrdict(dict):
    def __getattr__(self, attr):
        try:
            return dict.__getitem__(self, attr)
        except KeyError:
            return None

printsome = lambda a: " ".join(map(str,a.flatten()[:5]))

################################################################################
#Helper funcs

def writeStatus(name, ferr, ssr, wdf, bicount, bimarg_model, couplings, 
                seqs, startseq, energies, alpha, ptrate, outdir, log):
    
    C = bimarg_model - indep_bimarg(bimarg_model)
    X = sum(couplings*C, axis=1)

    #print some details 
    disp = ["Start Seq: " + "".join([alpha[c] for c in startseq]),
            "{} Ferr: {: 9.7f}  SSR: {: 9.5f}  dX: {: 9.5f}".format(
                                                     name, ferr,ssr,sum(X)),
            "{} wdf: {: 9.7f}".format(name, wdf),
            "Bicounts: " + printsome(bicount) + '...',
            "Marginals: " + printsome(bimarg_model) + '...',
            "Couplings: " + printsome(couplings) + "...",
            "Energies: Lowest =  {}, Mean = {}".format(min(energies), 
                                                       mean(energies))]
    if ptrate != 0:
        disp.append("PT swap rate: {:.2f}%".format(ptrate*100))

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

def newtonStep(n, bimarg_target, gamma, pc, reg_param, gpus, log):
    # expects the back buffers to contain current couplings & bimarg,
    # will overwrite front buffers

    # calculate perturbed marginals
    for gpu in gpus:
        # note: updateJ should give same result on all GPUs
        # overwrites J front using bi back and J back
        if reg_param is not None:
            gpu.updateJ_weightfn(gamma, pc, reg_param)
        else:
            gpu.updateJ(gamma, pc)

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

import pseudocount
def resample_target(unicounts, gpus):
    L, nB = unicounts.shape
    N = sum(unicounts[0,:])

    unimarg = unicounts/float(N)
    bins = cumsum(unimarg, axis=1)
    unicountss = [bincount(searchsorted(cp, rand(N)), minlength=8)
                  for cp in bins]
    unimargss = array(unicountss)/float(N)
    pairs = [(i,j) for i in range(L-1) for j in range(i+1,L)]
    bimarg_indep_ss = array([outer(unimargss[i], unimargss[j]).flatten()
                             for i,j in pairs], dtype='<f4')
    bimarg_indep_ss = pseudocount.prior(bimarg_indep_ss, 0.001)

    for gpu in gpus:
        gpu.setBuf('bi target', bimarg_indep_ss)

def resampled_target(ff, N, gpus):
    ct = N*ff
    sample = dists.binom.rvs(N, dists.beta.rvs(1+ct, 1+N-ct))
    ff = sample/sum(sample, axis=1).astype('<f4')[:,newaxis]
    return pseudocount.prior(ff, 0.001).astype('<f4')

def iterNewton(param, gpus, log):
    gamma = gamma0 = param.gamma0
    newtonSteps = param.newtonSteps
    pc = param.pcdamping
    gammasteps = 16
    bimarg_target = param.bimarg

    #unicounts = load('unicount5000.npy')

    
    if param.regularize:
        reg_param = param.fn_lmbda
    else:
        reg_param = None

    # setup front and back buffers. Back buffers should contain last accepted
    # values, front buffers vonctain trial values.
    for gpu in gpus:
        gpu.calcEnergies('large', 'main')
        gpu.copyBuf('J main', 'J back')
        gpu.copyBuf('J main', 'J front')
        gpu.setBuf('bi main', bimarg_model)
        gpu.copyBuf('bi main', 'bi back')

    log("Local target: ", printsome(bimarg_target))
    log("Local optimization:")
    
    # newton updates
    n = 1
    lastSSR = inf
    nrejects = 0
    for i in range(newtonSteps): 
        #resample_target(unicounts, gpus)
        bimarg_target_noise = resampled_target(bimarg_target, 1304, gpus)
        for gpu in gpus:
            gpu.setBuf('bi target', bimarg_target_noise)

        # increase gamma every gammasteps steps
        if 0 and i != 0 and i % gammasteps == 0:
            if nrejects == gammasteps:
                log("Too many ssr increases. Stopping newton updates.")
                break
            gamma = gamma*2
            log("Increasing gamma to {}".format(gamma))
            nrejects = 0
        
        # do newton step
        ssr, bimarg_model = newtonStep(n, bimarg_target, gamma, pc, reg_param,
                                       gpus, log)

        # accept move if ssr decreases, reject otherwise
        if True or ssr <= lastSSR:  # accept move
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
        gpu.calcBicounts('large')
    res = readGPUbufs(['bicount', 'seq large'], gpus)
    bicount, seqs = sumarr(res[1]), res[2]
    bimarg = bicount.astype(float32)/float32(sum(bicount[0,:]))
    
    #store initial setup
    mkdir_p(os.path.join(outdir, 'preopt'))
    log("Unweighted Marginals: ", printsome(bimarg))
    save(os.path.join(outdir, 'preopt', 'initbimarg'), bimarg)
    save(os.path.join(outdir, 'preopt', 'initJ'), couplings)
    save(os.path.join(outdir, 'preopt', 'initBicont'), bicount)
    for n,s in enumerate(seqs):
        seqload.writeSeqs(os.path.join(outdir, 'preopt', 'seqs-'+str(n)), 
                          s, alpha)
    
    topbi = bimarg_target > 0.01
    ferr = mean((abs(bimarg_target - bimarg)/bimarg_target)[topbi])
    ssr = sum((bimarg_target - bimarg)**2)
    wdf = sum(bimarg_target*abs(bimarg_target - bimarg))
    log("S Ferr: {: 9.7f}  SSR: {: 9.5f}  wDf: {: 9.5f}".format(ferr,ssr,wdf))

    #modify couplings a little
    couplings, bimarg_p = iterNewton(param, bimarg, gpus, log)
    save(os.path.join(outdir, 'preopt', 'perturbedbimarg'), bimarg_p)
    save(os.path.join(outdir, 'preopt', 'perturbedJ'), couplings)

def swapTempsCPU(gpus, N):
    # CPU implementation of PT swap

    for gpu in gpus:
        gpu.calcEnergies('small', 'main')
    es, Bs = map(concatenate, readGPUbufs(['E small', 'Bs'], gpus))
    ns = len(es)

    nswaps = 0
    for n in range(N):
        # randomly select position i
        i = np.random.randint(ns)
        
        # the gibbs sample to get position j
        delta = exp(min((es[i] - es)*(Bs[i] - Bs), 0))
        cumsum(delta, out=delta)
        j = searchsorted(delta, rand()*delta[-1])
        
        if Bs[i] != Bs[j]:
            Bs[i], Bs[j] = Bs[j], Bs[i] 
            nswaps += 1

    Bs = split(Bs, len(gpus)) 
    for B,gpu in zip(Bs, gpus):
        gpu.setBuf('Bs', B)

    return nswaps, Bs

def swapTempsCPU(gpus, N):
    # CPU implementation of PT swap

    for gpu in gpus:
        gpu.calcEnergies('small', 'main')
    es, Bs = map(concatenate, readGPUbufs(['E small', 'Bs'], gpus))
    ns = len(es)

    order = argsort(es)
    es = es[order]
    Bs = Bs[order]

    deEvn = es[ :-1:2] - es[1::2]
    deOdd = es[1:-1:2] - es[2::2]

    nswaps = 0
    for n in range(N):
        # swap even
        delta1 = deEven*(bs[:-1:2] - bs[1::2])
        swap = 2*where(exp(min(delta1,0)) < rand(ns-1))[0]
        tmp = Bs[swap]
        Bs[swap] = Bs[swap+1]
        Bs[swap+1] = tmp

        # swap odd
        delta2 = deOdd*(bs[1:-1:2] - bs[2::2])
        swap = 2*where(exp(min(delta2,0)) < rand(ns-1))[0] + 1
        tmp = Bs[swap]
        Bs[swap] = Bs[swap+1]
        Bs[swap+1] = tmp
    
    Bs[order] = Bs.copy()
    Bs = split(Bs, len(gpus)) 
    for B,gpu in zip(Bs, gpus):
        gpu.setBuf('Bs', B)

    return nswaps, Bs

def swapTempsGPU(gpus, N):
    # GPU implementation of PT swap.

    for gpu in gpus:
        gpu.calcEnergies('small', 'main')
    es, Bs = map(concatenate, readGPUbufs(['E small', 'Bs'], gpus))

    gpus[0].setBuf('E pt', es)
    gpus[0].setBuf('B pt', Bs)
    gpus[0].PTswap(N)
    Bs = readGPUbufs(['B pt'], gpus)[0]

    for B,gpu in zip(split(Bs, len(gpus)), gpus):
        gpu.setBuf('Bs', B)

    return 0, Bs

swapTemps = swapTempsGPU

# probably should just have a second version of this with PT, instead of all the ifs
def runMCMC(gpus, startseq, couplings, runName, param):
    nloop = param.equiltime
    nsamples = param.nsamples
    nsampleloops = param.sampletime
    trackequil = param.trackequil
    outdir = param.outdir
    # assumes small sequence buffer is already filled

    if param.tempering is not None:
        B0 = param.tempering[0]

    #get ready for MCMC
    for gpu in gpus:
        gpu.setBuf('J main', couplings)
    
    #equilibration MCMC
    meanswapfreq = 0
    if trackequil == 0:
        #keep nloop iterator on outside to avoid filling queue with only 1 gpu
        for i in range(nloop):
            for gpu in gpus:
                gpu.runMCMC()

            if param.tempering is not None:
                nswp, Bs = swapTemps(gpus, param.nswaps)
                meanswapfreq += float(nswp)/param.nswaps
    else:
        #note: sync necessary with trackequil (may slightly affect performance)
        mkdir_p(os.path.join(outdir, runName, 'equilibration'))
        for j in range(nloop/trackequil):
            for i in range(trackequil):
                for gpu in gpus:
                    gpu.runMCMC()

                if param.tempering is not None:
                    nswp, Bs = swapTemps(gpus, param.nswaps)
                    meanswapfreq += float(nswp)/param.nswaps

            if param.tempering is not None:
                for B,gpu in zip(Bs,gpus):
                    gpu.markSeqs(B == B0)

            for gpu in gpus:
                gpu.calcMarkedBicounts()
            bicounts = sumarr(readGPUbufs(['bicount'], gpus)[0])
            bimarg_model = bicounts.astype(float32)/float32(sum(bicounts[0,:]))
            save(os.path.join(outdir, runName, 
                 'equilibration', 'bimarg_{}'.format(j)), bimarg_model)

    #post-equilibration samples
    for gpu in gpus:
        gpu.clearLargeSeqs()
        gpu.storeMarkedSeqs() #save seqs from smallbuf to largebuf
    for j in range(1,nsamples):
        for i in range(nsampleloops):
            for gpu in gpus:
                gpu.runMCMC()

            if param.tempering is not None:
                nswp, Bs = swapTemps(gpus, param.nswaps)

        if param.tempering is not None:
            for B,gpu in zip(Bs,gpus):
                gpu.markSeqs(B == B0)
                gpu.storeMarkedSeqs()
        else:
            for gpu in gpus:
                gpu.storeSeqs()
    
    #process results
    for gpu in gpus:
        gpu.calcBicounts('large')
        gpu.calcEnergies('large', 'main')
    res = readGPUbufs(['bicount', 'E large', 'seq large'], gpus)
    bicount = sumarr(res[0])
    # assert sum(bicount, axis=1) are all equal here
    bimarg_model = bicount.astype(float32)/float32(sum(bicount[0,:]))
    sampledenergies, sampledseqs = concatenate(res[1]), res[2]
    
    ptrate = meanswapfreq/nloop
    return bimarg_model, bicount, sampledenergies, sampledseqs, ptrate

def MCMCstep(runName, startseq, couplings, param, gpus, log):
    outdir = param.outdir
    alpha, L, nB = param.alpha, param.L, param.nB
    bimarg_target = param.bimarg

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
    
    if param.resetseqs:
        for gpu in gpus:
            gpu.fillSeqs(startseq)

    (bimarg_model, 
     bicount, 
     sampledenergies, 
     sampledseqs,
     ptrate) = runMCMC(gpus, startseq, couplings, runName, param)

    #get summary statistics and output them
    topbi = bimarg_target > 0.01
    ferr = mean((abs(bimarg_target - bimarg_model)/bimarg_target)[topbi])
    ssr = sum((bimarg_target - bimarg_model)**2)
    wdf = sum(bimarg_target*abs(bimarg_target - bimarg_model))
    writeStatus(runName, ferr, ssr, wdf, bicount, bimarg_model, 
                couplings, sampledseqs, startseq, sampledenergies, 
                alpha, ptrate, outdir, log)
    
    #compute new J using local newton updates (in-place on GPU)
    couplings, bimarg_p = iterNewton(param, bimarg_model, gpus, log)
    save(os.path.join(outdir, runName, 'predictedBimarg'), bimarg_p)

    #choose seed sequence for next round
    rseq_ind = numpy.random.randint(0, len(sampledenergies))
    for s in sampledseqs:
        if rseq_ind < s.shape[0]:
            rseq = s[rseq_ind]
            break
        rseq_ind -= s.shape[0]

    return rseq, couplings

def newtonMCMC(param, gpus, log):
    startseq = param.startseq
    couplings = param.couplings

    if startseq is None:
        raise Exception("Error: Potts inference requires a starting sequence")
    
    if param.tempering is not None:
        if param.nwalkers % len(param.tempering) != 0:
            raise Exception("# of temperatures must evenly divide # walkers")
        B0 = param.tempering[0]
        Bs = concatenate([
             ones(param.nwalkers/len(param.tempering), dtype='f4')/t 
                          for t in param.tempering])
        shuffle(Bs)
        Bs = split(Bs, len(gpus)) 
        for B,gpu in zip(Bs, gpus):
            gpu.setBuf('Bs', B)
            gpu.markSeqs(B == B0)
    
    # pre-optimization
    if param.preopt:
        if param.seqs is None:
            raise Exception("Error: sequence buffers must be filled for "
                            "pre-optimization") 
                            #this doesn't actually check that....
        preOpt(param, gpus, log)
    elif param.preequiltime != 0:
        log("Pre-equilibration for {} steps...".format(param.preequiltime))
        log("(Re-centering gauge of couplings)")
        L, nB = param.L, param.nB
        couplings = fieldlessGaugeEven(zeros((L,nB)), couplings)[1]
        if param.resetseqs:
            for gpu in gpus:
                gpu.fillSeqs(startseq)
        for gpu in gpus:
            gpu.setBuf('J main', couplings)
        for i in range(param.preequiltime):
            for gpu in gpus:
                gpu.runMCMC()
            if param.tempering is not None:
                swapTemps(gpus, param.nswaps)
        log("Preequilibration done.")
    else:
        log("No Pre-optimization")

    
    # copy target bimarg to gpus
    for gpu in gpus:
        gpu.setBuf('bi target', param.bimarg)
    
    # setup up regularization if needed
    if param.regularize:
        for gpu in gpus:
            gpu.setBuf('Creg', param.Creg)
        
    
    # solve using newton-MCMC
    for i in range(param.mcmcsteps):
        runname = 'run_{}'.format(i)
        startseq, couplings = MCMCstep(runname, startseq, couplings, 
                                       param, gpus, log)

#!/usr/bin/env python2
#
#Copyright 2016 Allan Haldane.

#This file is part of IvoGPU.

#IvoGPU is free software: you can redistribute it and/or modify
#it under the terms of the GNU General Public License as published by
#the Free Software Foundation, version 3 of the License.

#IvoGPU is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#GNU General Public License for more details.

#You should have received a copy of the GNU General Public License
#along with IvoGPU.  If not, see <http://www.gnu.org/licenses/>.

#Contact: allan.haldane _AT_ gmail.com
from __future__ import print_function
from scipy import *
import scipy
import scipy.stats.distributions as dists
from scipy.stats import pearsonr
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
from IvoGPU import generateSequences

def unimarg(bimarg):
    L, q = seqsize_from_param_shape(bimarg.shape)
    ff = bimarg.reshape((L*(L-1)/2,q,q))
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
    q = int(sqrt(shape[1]) + 0.5) 
    return L, q

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

def printstats(name, bicount, bimarg_target, bimarg_model, couplings,
               energies, e_rho, ptinfo):
    topbi = bimarg_target > 0.01
    ferr = mean((abs(bimarg_target - bimarg_model)/bimarg_target)[topbi])
    ssr = sum((bimarg_target - bimarg_model)**2)
    wdf = sum(bimarg_target*abs(bimarg_target - bimarg_model))

    C = bimarg_model - indep_bimarg(bimarg_model)
    X = sum(couplings*C, axis=1)

    Co = bimarg_target - indep_bimarg(bimarg_target)
    Xo = sum(couplings*Co, axis=1)
    
    rho, rp = zip(*e_rho)
    rhostr = np.array2string(np.array(rho), precision=2, floatmode='fixed',
                             suppress_small=True)

    #print some details 
    disp = [("{} Ferr: {: 7.5f}  SSR: {: 7.3f}  dX: {: 7.3f} : {: 7.3f}"
            ).format(name, ferr, ssr, sum(X), sum(Xo)),
            "{} wdf: {: 9.7f}".format(name, wdf),
            "Bicounts: " + printsome(bicount) + '...',
            "Marginals: " + printsome(bimarg_model) + '...',
            "Couplings: " + printsome(couplings) + "...",
            "{} Energies: Lowest =  {}, Mean = {}".format(name,
                              min(energies), mean(energies)),
            "{} Energy Autocorrelation: {}".format(name, rhostr)]
    if ptinfo != None:
        disp.append("PT swap rate: {}".format(ptinfo[1]))

    return "\n".join(disp)

def writeStatus(name, bimarg_target, bicount, bimarg_model, couplings, 
                seqs_large, seqs, energies, alpha, e_rho, ptinfo, outdir, log):

    dispstr = printstats(name, bicount, bimarg_target, bimarg_model, couplings,
                         energies, e_rho, ptinfo)
    with open(os.path.join(outdir, name, 'info.txt'), 'wt') as f:
        f.write(dispstr)

    #save current state to file
    savetxt(os.path.join(outdir, name, 'bicounts'), bicount, fmt='%d')
    save(os.path.join(outdir, name, 'bimarg'), bimarg_model)
    save(os.path.join(outdir, name, 'energies'), energies)
    for n,seqbuf in enumerate(seqs_large):
        seqload.writeSeqs(os.path.join(outdir, name, 'seqs_large-{}'.format(n)), 
                          seqbuf, alpha, zipf=True)
    for n,seqbuf in enumerate(seqs):
        seqload.writeSeqs(os.path.join(outdir, name, 'seqs-{}'.format(n)), 
                          seqbuf, alpha, zipf=True)
    if ptinfo != None:
        for n,B in enumerate(ptinfo[0]):
            save(os.path.join(outdir, 'Bs-{}'.format(n)), B)

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

def newtonStep(n, bimarg_target, gamma, pc, reg, gpus, log):
    # expects the back buffers to contain current couplings & bimarg,
    # will overwrite front buffers

    # calculate perturbed marginals
    for gpu in gpus:
        # note: updateJ should give same result on all GPUs
        # overwrites J front using bi back and J back
        if reg[0] == 'Creg':
            gpu.updateJ_weightfn(gamma, pc)
        elif reg[0] == 'l2z':
            lh, lJ = reg[1]
            gpu.updateJ_l2z(gamma, pc, lh, lJ)
        elif reg[0] == 'Lstep':
            gpu.updateJ_Lstep(gamma, pc)
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
    #save('bimodel'+str(n), bimarg_model)

    if isinf(Neff) or Neff == 0:
        raise Exception("Error: Divergence. Decrease gamma or increase "
                        "pc-damping")
    
    # dump profiling info if profiling is turned on
    for gpu in gpus:
        gpu.logProfile()

    return SSR, bimarg_model

import pseudocount
def resample_target(unicounts, gpus):
    L, q = unicounts.shape
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

def singleNewton(bimarg, gamma, pc, gpus):
    gpus[0].setBuf('bi back', bimarg)
    gpus[0].copyBuf('J main', 'J back')
    # note: updateJ should give same result on all GPUs
    # overwrites J front using bi back and J back
    gpus[0].updateJ(gamma, pc)
    return gpus[0].getBuf('J front').read()

def resampled_target(ff, N, gpus):
    ct = N*ff
    sample = dists.binom.rvs(N, dists.beta.rvs(1+ct, 1+N-ct))
    ff = sample/sum(sample, axis=1).astype('<f4')[:,newaxis]
    return pseudocount.prior(ff, 0.001).astype('<f4')

def iterNewton(param, bimarg_model, gpus, log):
    gamma = gamma0 = param.gamma0
    newtonSteps = param.newtonSteps
    pc = param.pcdamping
    gammasteps = 16
    bimarg_target = param.bimarg
    reg = (param.reg, param.regarg)

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
        if param.noiseN is not None:
            bimarg_target_noise = resampled_target(bimarg_target, 
                                                   param.noiseN, gpus)
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
        ssr, bimarg_model = newtonStep(n, bimarg_target, gamma, pc, reg,
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

    log("Pre-Optimization using large seq buffer:")
    
    for gpu in gpus:
        gpu.setBuf('J main', couplings)
        gpu.calcEnergies('large', 'main')
        gpu.calcBicounts('large')
    res = readGPUbufs(['bicount', 'seq large', 'E large'], gpus)
    bicount, seqs, energies = sumarr(res[0]), res[1], concatenate(res[2])
    bimarg = bicount.astype(float32)/float32(sum(bicount[0,:]))
    
    #store initial setup
    mkdir_p(os.path.join(outdir, 'preopt'))
    log("Unweighted Marginals: ", printsome(bimarg))
    save(os.path.join(outdir, 'preopt', 'initbimarg'), bimarg)
    save(os.path.join(outdir, 'preopt', 'initJ'), couplings)
    save(os.path.join(outdir, 'preopt', 'initBicount'), bicount)
    for n,s in enumerate(seqs):
        seqload.writeSeqs(os.path.join(outdir, 'preopt', 'seqs-'+str(n)+'.bz2'),
                          s, alpha, zipf=True)
    
    topbi = bimarg_target > 0.01
    ferr = mean((abs(bimarg_target - bimarg)/bimarg_target)[topbi])
    ssr = sum((bimarg_target - bimarg)**2)
    wdf = sum(bimarg_target*abs(bimarg_target - bimarg))
    log("S Ferr: {: 9.7f}  SSR: {: 9.5f}  wDf: {: 9.5f}".format(ferr,ssr,wdf))

    #modify couplings a little
    if param.newtonSteps != 1:
        couplings, bimarg_p = iterNewton(param, bimarg, gpus, log)
        save(os.path.join(outdir, 'preopt', 'perturbedbimarg'), bimarg_p)
        save(os.path.join(outdir, 'preopt', 'perturbedJ'), couplings)
    else:
        log("Performing single newton update step")
        couplings = singleNewton(bimarg, param.gamma0,
                                 param.pcdamping, gpus)

    log(printstats('preopt', bicount, bimarg_target, bimarg,
                   couplings, energies, None, None))

    return couplings


def swapTemps(gpus, dummy, N):
    # CPU implementation of PT swap
    t1 = time.time()

    for gpu in gpus:
        gpu.calcEnergies('main', 'main')
    es, Bs = map(concatenate, readGPUbufs(['E main', 'Bs'], gpus))
    ns = len(es)

    Bs_orig = Bs.copy()
    #r1 = logaddexp.reduce(-Bs*es)/len(Bs)
    
    # swap consecutive replicas, where consecutive is in E order
    order = argsort(es)
    #order = arange(len(es))
    es = es[order]
    Bs = Bs[order]

    deEvn = es[ :-1:2] - es[1::2]
    deOdd = es[1:-1:2] - es[2::2]
    
    # views of even and odd elements
    eBs1, eBs2 = Bs[:-1:2], Bs[1::2]
    oBs1, oBs2 = Bs[1:-1:2], Bs[2::2]

    def swap(a, b, dE):
        ind_diff = where(a != b)[0]
        delta = dE[ind_diff]*(a[ind_diff] - b[ind_diff])
        ind_lz = where(delta < 0)[0]
        sind = ones(len(delta), dtype='bool')
        sind[ind_lz] = exp(delta[ind_lz]) > rand(len(ind_lz))
        sind = ind_diff[sind]

        a[sind], b[sind] = b[sind], a[sind]
    
    for n in range(N):
        swap(eBs1, eBs2, deEvn)
        swap(oBs1, oBs2, deOdd)
    
    # get back to original order
    Bs[order] = Bs.copy()

    r = sum(Bs != Bs_orig)/float(len(Bs))
    #r2 = logaddexp.reduce(-Bs*es)/len(Bs)

    Bs = split(Bs, len(gpus)) 
    for B,gpu in zip(Bs, gpus):
        gpu.setBuf('Bs', B)

    t2 = time.time()
    #print(t2-t1)
    return Bs, r

def track_main_bufs(gpus, savedir=None, step=None):
    for gpu in gpus:
        gpu.calcBicounts('main')
        gpu.calcEnergies('main', 'main')
    bicounts, energies = readGPUbufs(['bicount', 'E main'], gpus)
    bicounts = sumarr(bicounts)
    bimarg_model = bicounts.astype('f4')/float32(sum(bicounts[0,:]))
    energies = concatenate(energies)
    
    if savedir:
        save(os.path.join(savedir, 'bimarg_{}'.format(step)), bimarg_model)
        save(os.path.join(savedir, 'energies_{}'.format(step)), energies)
    return energies, bimarg_model

def runMCMC(gpus, couplings, runName, param, log):
    nloop = param.equiltime
    nsamples = param.nsamples
    nsampleloops = param.sampletime
    trackequil = param.trackequil
    outdir = param.outdir
    # assumes small sequence buffer is already filled

    #get ready for MCMC
    for gpu in gpus:
        gpu.setBuf('J main', couplings)
    
    #equilibration MCMC
    if nloop == 'auto':
        equil_dir = os.path.join(outdir, runName, 'equilibration')
        mkdir_p(equil_dir)

        loops = 32
        for i in range(loops):
            for gpu in gpus:
                gpu.runMCMC()
        step = loops

        equil_e = []
        while True:
            for i in range(loops):
                for gpu in gpus:
                    gpu.runMCMC()

            step += loops
            energies, _ = track_main_bufs(gpus, equil_dir, step=step)
            equil_e.append(energies)

            if len(equil_e) >= 3:
                r1, p1 = pearsonr(equil_e[-1], equil_e[-2])
                r2, p2 = pearsonr(equil_e[-1], equil_e[-3])

                fmt = "({:.3f}, {:.2g})".format
                rstr = "Step {}, r=cur:{} prev:{}".format(
                        step, fmt(r1, p1), fmt(r2, p2))

                if p1 > 0.02 and p2 > 0.02:
                    log(rstr + ". Equilibrated.")
                    break
            else:
                rstr = "Step {}".format(step)

            loops = loops*2
            log(rstr + ". Continuing equilibration.")

        e_rho = [pearsonr(ei, equil_e[-1]) for ei in equil_e]

    elif trackequil == 0:
        #keep nloop iterator on outside to avoid filling queue with only 1 gpu
        for i in range(nloop):
            for gpu in gpus:
                gpu.runMCMC()

        e_rho = None
    else:
        #note: sync necessary with trackequil (may slightly affect performance)
        mkdir_p(os.path.join(outdir, runName, 'equilibration'))
        equil_e = []
        for j in range(nloop/trackequil):
            for i in range(trackequil):
                for gpu in gpus:
                    gpu.runMCMC()
            energies, _ = track_main_bufs(gpus, equil_dir, step=step)
            equil_e.append(energies)

        # track how well different walkers are equilibrated. Should go to 0
        e_rho = [pearsonr(ei, equil_e[-1]) for ei in equil_e]

    #post-equilibration samples
    for gpu in gpus:
        gpu.clearLargeSeqs()
        gpu.storeSeqs() #save seqs from smallbuf to largebuf
    for j in range(1,nsamples):
        for i in range(nsampleloops):
            for gpu in gpus:
                gpu.runMCMC()
        for gpu in gpus:
            gpu.storeSeqs()
    
    #process results
    for gpu in gpus:
        gpu.calcBicounts('large')
        gpu.calcEnergies('large', 'main')
    res = readGPUbufs(['bicount', 'E large'], gpus)
    bicount = sumarr(res[0])
    # assert sum(bicount, axis=1) are all equal here
    bimarg_model = bicount.astype(float32)/float32(sum(bicount[0,:]))
    sampledenergies = concatenate(res[1])
    
    return bimarg_model, bicount, sampledenergies, e_rho, None

def runMCMC_tempered(gpus, couplings, runName, param, log):
    nloop = param.equiltime
    nsamples = param.nsamples
    nsampleloops = param.sampletime
    trackequil = param.trackequil
    outdir = param.outdir
    # assumes small sequence buffer is already filled

    B0 = np.max(param.tempering)

    #get ready for MCMC
    for gpu in gpus:
        gpu.setBuf('J main', couplings)
    
    #equilibration MCMC
    if trackequil == 0:
        #keep nloop iterator on outside to avoid filling queue with only 1 gpu
        for i in range(nloop):
            for gpu in gpus:
                gpu.runMCMC()
            Bs,r = swapTemps(gpus, param.tempering, param.nswaps)
    else:
        #note: sync necessary with trackequil (may slightly affect performance)
        mkdir_p(os.path.join(outdir, runName, 'equilibration'))
        equil_e = []
        for j in range(nloop/trackequil):
            for i in range(trackequil):
                for gpu in gpus:
                    gpu.runMCMC()
                Bs,r = swapTemps(gpus, param.tempering, param.nswaps)
            #for B,gpu in zip(Bs,gpus):
            #    gpu.markSeqs(B == B0)
            #    gpu.calcMarkedBicounts()
            #bicounts = sumarr(readGPUbufs(['bicount'], gpus)[0])
            #bimarg_model = bicounts.astype(float32)/float32(sum(bicounts[0,:]))
            #save(os.path.join(outdir, runName, 
            #     'equilibration', 'bimarg_{}'.format(j)), bimarg_model)

            save(os.path.join(outdir, runName, 
                 'equilibration', 'Bs_{}'.format(j)), concatenate(Bs))
            energies = concatenate(readGPUbufs(['E main'], gpus)[0])
            save(os.path.join(outdir, runName, 
                 'equilibration', 'energies_{}'.format(j)), energies)
            equil_e.append(energies)

        # track how well different walkers are equilibrated. Should go to 0
        # Note: Do we expect this to go to 0 with PT? SHould we only
        # be looking at walkers which ate at the lowest temperature?
        # No, this needs to be changed... it gives large correlations when there should
        # be none. May be because we did not reach the infinite swap limit,
        # so even if the walker distribution is equilibrated the PT swapping is not.
        e_rho = [pearsonr(ei, equil_e[-1])[0] for ei in equil_e]

    for B,gpu in zip(Bs,gpus):
        gpu.markSeqs(B == B0)

    #post-equilibration samples
    for gpu in gpus:
        gpu.clearLargeSeqs()
        gpu.storeMarkedSeqs() #save seqs from smallbuf to largebuf
    for j in range(1,nsamples):
        for i in range(nsampleloops):
            for gpu in gpus:
                gpu.runMCMC()

            Bs,r = swapTemps(gpus, param.tempering, param.nswaps)
        for B,gpu in zip(Bs,gpus):
            gpu.markSeqs(B == B0)
            gpu.storeMarkedSeqs()
    
    #process results
    for gpu in gpus:
        gpu.calcBicounts('large')
        gpu.calcEnergies('large', 'main')
    res = readGPUbufs(['bicount', 'E large'], gpus)
    bicount = sumarr(res[0])
    # assert sum(bicount, axis=1) are all equal here
    bimarg_model = bicount.astype(float32)/float32(sum(bicount[0,:]))
    sampledenergies = concatenate(res[1])
    
    return bimarg_model, bicount, sampledenergies, e_rho, (Bs, r)

def MCMCstep(runName, couplings, param, gpus, log):
    outdir = param.outdir
    alpha, L, q = param.alpha, param.L, param.q
    bimarg_target = param.bimarg

    log("")
    log("Gradient Descent step {}".format(runName))

    #re-distribute energy among couplings
    #(not really needed, but makes nicer output and might prevent
    # numerical inaccuracy, but also shifts all seq energies)
    #log("(Re-centering gauge of couplings)")
    #couplings = fieldlessGaugeEven(zeros((L,q)), couplings)[1]

    mkdir_p(os.path.join(outdir, runName))
    save(os.path.join(outdir, runName, 'J'), couplings)
    
    MCMC_func = runMCMC
    if param.tempering is not None:
        MCMC_func = runMCMC_tempered

    log("Equilibrating MCMC chains...")
    (bimarg_model, 
     bicount, 
     sampledenergies,
     e_rho,
     ptinfo) = MCMC_func(gpus, couplings, runName, param, log)

    seq_large, seqs = readGPUbufs(['seq large', 'seq main'], gpus)

    #get summary statistics and output them
    writeStatus(runName, bimarg_target, bicount, bimarg_model, 
                couplings, seq_large, seqs, sampledenergies, 
                alpha, e_rho, ptinfo, outdir, log)

    if param.tempering is not None:
        e, b = readGPUbufs(['E main', 'Bs'], gpus)
        save(os.path.join(outdir, runName, 'walker_Es'), concatenate(e))
        save(os.path.join(outdir, runName, 'walker_Bs'), concatenate(b))
    
    #compute new J using local newton updates (in-place on GPU)
    if param.newtonSteps != 1:
        newcouplings, bimarg_p = iterNewton(param, bimarg_model, gpus, log)
        save(os.path.join(outdir, runName, 'predictedBimarg'), bimarg_p)
        save(os.path.join(outdir, runName, 'predictedJ'), newcouplings)
    else:
        log("Performing single newton update step")
        newcouplings = singleNewton(bimarg_model, param.gamma0,
                                    param.pcdamping, gpus)

    return seqs, newcouplings

def newtonMCMC(param, gpus, log):
    couplings = param.couplings
    
    # copy target bimarg to gpus
    for gpu in gpus:
        gpu.setBuf('bi target', param.bimarg)

    if param.reseed.startswith('single') and param.seedseq is None:
        raise Exception("Error: resetseq option requires a starting sequence")
    
    if param.tempering is not None:
        if param.nwalkers % len(param.tempering) != 0:
            raise Exception("# of temperatures must evenly divide # walkers")
        B0 = np.max(param.tempering)
        Bs = concatenate([
             full(param.nwalkers/len(param.tempering), b, dtype='f4') 
                          for b in param.tempering])
        Bs = split(Bs, len(gpus)) 
        for B,gpu in zip(Bs, gpus):
            gpu.setBuf('Bs', B)
            gpu.markSeqs(B == B0)
    
    # pre-optimization
    if param.preopt:
        if param.seqs_large is None:
            raise Exception("Error: sequence buffers must be filled for "
                            "pre-optimization") 
        couplings = preOpt(param, gpus, log)
    elif param.preequiltime:
        log("Pre-equilibration for {} steps...".format(param.preequiltime))
        log("(Re-centering gauge of couplings)")
        L, q = param.L, param.q
        couplings = fieldlessGaugeEven(zeros((L,q)), couplings)[1]
        if param.resetseqs:
            for gpu in gpus:
                gpu.fillSeqs(seedseq)
        for gpu in gpus:
            gpu.setBuf('J main', couplings)
        for i in range(param.preequiltime):
            for gpu in gpus:
                gpu.runMCMC()
            if param.tempering is not None:
                Bs, r = swapTemps(gpus, param.tempering, param.nswaps)
        log("Preequilibration done.")
        if param.tempering is not None:
            log("Final PT swap rate: {}".format(r))
    else:
        log("No Pre-optimization")

    # setup up regularization if needed
    if param.reg == 'Creg':
        for gpu in gpus:
            gpu.setBuf('Creg', param.regarg)

    # solve using newton-MCMC
    for i in range(param.mcmcsteps):
        runname = 'run_{}'.format(i)
        
        if param.reseed.startswith('single'):
            mkdir_p(os.path.join(param.outdir, runname))
            with open(os.path.join(param.outdir,runname, 'seedseq'), 'wt') as f:
                f.write("".join(param.alpha[c] for c in param.seedseq))
            for gpu in gpus:
                gpu.fillSeqs(param.seedseq)
        elif param.reseed == 'independent': 
            for gpu in gpus:
                indep_seqs = generateSequences('independent', param.L, param.q,
                                            gpu.nseq['main'], param.bimarg, log)
                gpu.setBuf('seq main', indep_seqs)

        seqs, couplings = MCMCstep(runname, couplings, param, gpus, log)

        if param.reseed == 'single_random':
            #choose random seed sequence from the final sequences for next round
            nseq = sum(s.shape[0] for s in seqs)
            rseq = None
            rseq_ind = numpy.random.randint(0, nseq)
            for s in seqs:
                if rseq_ind < s.shape[0]:
                    rseq = s[rseq_ind]
                    break
                rseq_ind -= s.shape[0]
            param['seedseq'] = rseq
        if param.reseed == 'single_best':
            raise Exception("Not implemented yet")
        if param.reseed == 'single_indep':
            seq = generateSequences('independent', param.L, param.q, 1,
                                    param.bimarg, log)[0]
            param['seedseq'] = seq


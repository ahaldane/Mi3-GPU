#!/usr/bin/env python
#
#Copyright 2018 Allan Haldane.

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
import numpy as np
from scipy.stats import pearsonr, dirichlet, spearmanr
from numpy.random import randint
import pyopencl as cl
import sys, os, errno, glob, argparse, time
from utils.changeGauge import fieldlessGaugeEven
from utils.seqload import writeSeqs, loadSeqs
from utils import printsome, getLq, indep_bimarg, unimarg

from mcmcGPU import readGPUbufs, merge_device_bimarg
from IvoGPU import generateSequences

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


################################################################################
#Helper funcs

def printstats(name, bicount, bimarg_target, bimarg_model, couplings,
               energies, e_rho, ptinfo):
    topbi = bimarg_target > 0.01
    with np.errstate(divide='ignore', invalid='ignore'):
        rel_err = (np.abs(bimarg_target - bimarg_model)/bimarg_target)
        ferr = np.mean(rel_err[topbi])
    ssr = np.sum((bimarg_target - bimarg_model)**2)

    C = bimarg_model - indep_bimarg(bimarg_model)
    X = np.sum(couplings*C, axis=1)

    Co = bimarg_target - indep_bimarg(bimarg_target)
    Xo = np.sum(couplings*Co, axis=1)
    
    rhostr = '(none)'
    if e_rho is not None:
        rho, rp = zip(*e_rho)
        rhostr = np.array2string(np.array(rho), precision=2, floatmode='fixed',
                                 suppress_small=True)[1:-1]

    #print some details
    disp = """\
{name} Ferr: {ferr: 7.5f}  SSR: {ssr: 7.3f}  dX: {dX: 7.3f} : {dXo: 7.3f}
{name} Bicounts: {bicounts} ...
{name} Marginals: {bimarg} ...
{name} Couplings: {couplings} ...
{name} Energies: Lowest = {lowE:.4f}, Mean = {meanE:.4f}
{name} Energy Autocorrelation vs time: {rhos}""".format(
        name=name, ferr=ferr, ssr=ssr, dX=np.sum(X), dXo=np.sum(Xo),
        bicounts=printsome(bicount), bimarg=printsome(bimarg_model),
        couplings=printsome(couplings), lowE=min(energies),
        meanE=np.mean(energies), rhos=rhostr)

    if ptinfo != None:
        disp += "\n{} PT swap rate: {}".format(name, ptinfo[1])

    return disp

def writeStatus(name, bimarg_target, bicount, bimarg_model, couplings,
                seqs, energies, alpha, e_rho, ptinfo, outdir, log):

    dispstr = printstats(name, bicount, bimarg_target, bimarg_model, couplings,
                         energies, e_rho, ptinfo)
    with open(os.path.join(outdir, name, 'info.txt'), 'wt') as f:
        f.write(dispstr)

    #save current state to file
    np.savetxt(os.path.join(outdir, name, 'bicounts'), bicount, fmt='%d')
    np.save(os.path.join(outdir, name, 'bimarg'), bimarg_model)
    np.save(os.path.join(outdir, name, 'energies'), energies)
    for n,seqbuf in enumerate(seqs):
        fn = os.path.join(outdir, name, 'seqs-{}'.format(n))
        writeSeqs(fn, seqbuf, alpha, zipf=True)
    if ptinfo != None:
        for n,B in enumerate(ptinfo[0]):
            np.save(os.path.join(outdir, 'Bs-{}'.format(n)), B)

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

def singleNewton(bimarg, gamma, param, gpus):
    pc = param.pcdamping

    gpus[0].setBuf('bi', bimarg)
    # note: updateJ should give same result on all GPUs
    # overwrites J front using bi back and J back
    if param.reg == 'l2z':
        lh, lJ = param.regarg
        gpus[0].updateJ_l2z(gamma, pc, lh, lJ)
    elif param.reg == 'X':
        gpus[0].updateJ_X(gamma, pc)
    elif param.reg == 'Xself':
        gpus[0].updateJ_Xself(gamma, pc)
    else:
        gpus[0].updateJ(gamma, pc)
    return gpus[0].getBuf('J').read()

def getNeff(w):
    # This corresponds to an effective N for the weighted average of N
    # bernoulli trials, eg X = (w1 X1 + w2 X2 + ...)/sum(w) for which we find
    # var(X) = p(1-p)/Neff, with Neff = N in the unweighted case.
    return (np.sum(w)**2)/np.sum(w**2)

def NewtonStatus(n, trialJ, weights, bimarg_model, bimarg_target, log):
    SSR = np.sum((bimarg_model.flatten() - bimarg_target.flatten())**2)
    
    # not clear what best Neff measure is. sum(weights)?
    Neff = getNeff(weights)
    log(("{}  ssr: {}  Neff: {:.1f} wspan: {:.3g}:{:.3g}").format(
         n, SSR, Neff, min(weights), max(weights)))
    log("    trialJ:", printsome(trialJ)), '...'
    log("    bimarg:", printsome(bimarg_model)), '...'
    log("   weights:", printsome(weights)), '...'

def iterNewton_multiGPU(param, bimarg_model, gpus, log):
    bimarg_target = param.bimarg
    gamma = param.gamma0
    newtonSteps = param.newtonSteps
    pc = param.pcdamping
    Nfrac = param.fracNeff
    N = param.nwalkers

    log("")
    log("Local optimization for {} steps:".format(newtonSteps))
    if param.reg:
        log("Using regularization {}".format(param.reg))

    if param.reg == 'l2z':
        updateJ = lambda gpu: gpu.updateJ_l2z(gamma, pc, *param.regarg)
    elif param.reg == 'X':
        updateJ = lambda gpu: gpu.updateJ_X(gamma, pc)
    elif param.reg == 'Xself':
        updateJ = lambda gpu: gpu.updateJ_Xself(gamma, pc)
    else:
        updateJ = lambda gpu: gpu.updateJ(gamma, pc)

    for gpu in gpus:
        gpu.calcEnergies('main')
        gpu.calcBicounts('main')
        gpu.bicounts_to_bimarg(gpu.nseq['main'])
    merge_device_bimarg(gpus)

    def getNewtonBufs():
        bi, J = readGPUbufs(['bi', 'J'], [gpus[0]])
        weights = np.concatenate(readGPUbufs(['weights'], gpus)[0])
        return bi[0], weights, J[0]

    # perform the newton update steps
    lastNeff = 2*N
    for i in range(newtonSteps):
        for gpu in gpus:
            updateJ(gpu) # should be the identical calculation on all gpus
            gpu.calcWeights('main')
            gpu.weightedMarg('main')
        merge_device_bimarg(gpus)

        weights = np.concatenate(readGPUbufs(['weights'], gpus)[0])
        Neff = getNeff(weights)
        if i%64 == 0 or abs(lastNeff - Neff)/N > 0.05 or Neff < Nfrac*N:
            log("J-step {: 5d}   Neff: {:.1f}   ({:.1f}% of {})".format(
                 i, Neff, Neff/N*100, N))
            lastNeff = Neff
        if Neff < Nfrac*N:
            log("Ending coupling updates because Neff/N < {:.2f}".format(Nfrac))
            break

    bimarg_model, weights, trialJ = getNewtonBufs()
    NewtonStatus(i, trialJ, weights, bimarg_model, bimarg_target, log)

    Neff = np.sum(weights)
    if not np.isfinite(Neff) or Neff == 0:
        raise Exception("Error: Divergence. Decrease gamma or increase "
                        "pc-damping")

    # dump profiling info if profiling is turned on
    for gpu in gpus:
        gpu.logProfile()

    return trialJ, bimarg_model

# The following calculation only uses the first GPU, and copies all the
# sequences to it.
def iterNewton_singleGPU(param, bimarg_model, gpus, log):
    bimarg_target = param.bimarg
    gamma = param.gamma0
    newtonSteps = param.newtonSteps
    pc = param.pcdamping
    gpu0 = gpus[0]
    Nfrac = param.fracNeff
    N = param.nwalkers

    log("")
    log("Local optimization for {} steps:".format(newtonSteps))
    if param.reg:
        log("Using regularization {}".format(param.reg))

    if param.reg == 'l2z':
        updateJ = lambda gpu: gpu.updateJ_l2z(gamma, pc, *param.regarg)
    elif param.reg == 'X':
        updateJ = lambda gpu: gpu.updateJ_X(gamma, pc)
    elif param.reg == 'Xself':
        updateJ = lambda gpu: gpu.updateJ_Xself(gamma, pc)
    else:
        updateJ = lambda gpu: gpu.updateJ(gamma, pc)

    # copy all sequences into gpu-0's large seq buffer
    seq_large = np.concatenate(readGPUbufs(['seq main'], gpus)[0], axis=0)
    gpu0.clearLargeSeqs()
    gpu0.storeSeqs(seq_large)
    gpu0.calcEnergies('large')
    gpu0.calcBicounts('large')
    gpu0.bicounts_to_bimarg(gpu0.nstoredseqs)

    def getNewtonBufs():
        bi, weights, J = readGPUbufs(['bi', 'weights large', 'J'], [gpu0])
        return bi[0], weights[0], J[0]

    # actually do coupling updates
    lastNeff = 2*N
    Ncutoff = 0.5
    for i in range(newtonSteps):
        updateJ(gpu0)
        gpu0.calcWeights('large')
        gpu0.weightedMarg('large')
        gpu0.renormalize_bimarg()

        weights = gpu0.getBuf('weights large').read()
        Neff = getNeff(weights)
        if i%64 == 0 or abs(lastNeff - Neff)/N > 0.05 or Neff < Nfrac*N:
            log("J-step {: 5d}   Neff: {:.1f}   ({:.1f}% of {})".format(
                 i, Neff, Neff/N*100, N))
            lastNeff = Neff
        if Neff < Nfrac*N:
            log("Ending coupling updates because Neff/N < {:.2f}".format(Nfrac))
            break

    bimarg_model, weights, trialJ = getNewtonBufs()
    NewtonStatus(i, trialJ, weights, bimarg_model, bimarg_target, log)

    Neff = np.sum(weights)
    if not np.isfinite(Neff) or Neff == 0:
        raise Exception("Error: Divergence. Decrease gamma or increase "
                        "pc-damping")

    # dump profiling info if profiling is turned on
    for gpu in gpus:
        gpu.logProfile()

    return trialJ, bimarg_model

import time
def iterNewton(param, bimarg_model, gpus, log):
    #return iterNewton_multiGPU(param, bimarg_model, gpus, log)
    #return iterNewton_singleGPU(param, bimarg_model, gpus, log)
    s = time.time()
    if param.multigpu:
        ret = iterNewton_multiGPU(param, bimarg_model, gpus, log)
    else:
        ret = iterNewton_singleGPU(param, bimarg_model, gpus, log)
    e = time.time()
    log("Total Newton-step running time: {:.1f} s".format(e-s))
    return ret


################################################################################

#pre-optimization steps
def preOpt(param, gpus, log):
    couplings = param.couplings
    outdir = param.outdir
    alpha = param.alpha
    bimarg_target = param.bimarg

    log("Pre-Optimization:")

    for gpu in gpus:
        gpu.setBuf('J', couplings)
        gpu.calcEnergies('main')
        gpu.calcBicounts('main')
    res = readGPUbufs(['bicount', 'seq main', 'E main'], gpus)
    bicount, seqs, energies = sumarr(res[0]), res[1], np.concatenate(res[2])
    bimarg = bicount.astype(np.float32)/np.float32(np.sum(bicount[0,:]))

    #store initial setup
    mkdir_p(os.path.join(outdir, 'preopt'))
    log("Unweighted Marginals: ", printsome(bimarg))
    np.save(os.path.join(outdir, 'preopt', 'initbimarg'), bimarg)
    np.save(os.path.join(outdir, 'preopt', 'initJ'), couplings)
    np.save(os.path.join(outdir, 'preopt', 'initBicount'), bicount)
    for n,s in enumerate(seqs):
        fn = os.path.join(outdir, 'preopt', 'seqs-'+str(n)+'.bz2')
        writeSeqs(fn, s, alpha, zipf=True)

    topbi = bimarg_target > 0.01
    with np.errstate(divide='ignore', invalid='ignore'):
        ferr = np.mean((abs(bimarg_target - bimarg)/bimarg_target)[topbi])
    ssr = np.sum((bimarg_target - bimarg)**2)
    wdf = np.sum(bimarg_target*abs(bimarg_target - bimarg))

    log(printstats('preopt', bicount, bimarg_target, bimarg,
                   couplings, energies, None, None))

    #modify couplings a little
    if param.newtonSteps != 1:
        couplings, bimarg_p = iterNewton(param, bimarg, gpus, log)
        np.save(os.path.join(outdir, 'preopt', 'perturbedbimarg'), bimarg_p)
        np.save(os.path.join(outdir, 'preopt', 'perturbedJ'), couplings)
    else:
        log("Performing single newton update step")
        couplings = singleNewton(bimarg, param.gamma0,
                                 param, gpus)

    return couplings

def rand_f32(size):
    """
    Generate random float32 values [0, 1), assuming little-endian IEEE binary32

    In the IEEE binary32 floating point format, such values all have the same
    exponent and only vary in the mantissa. So, we only need to generate the 23
    bit mantissa per uniform float32. It turns out in numpy it is fastest to
    generate 32 bits and then drop the first 9.
    """
    x = randint(np.uint32(2**32-1), size=size, dtype=np.uint32)
    shift, exppat = np.uint32(9), np.uint32(0x3F800000)
    return ((x >> shift) | exppat).view('f4') - np.float32(1.0)

def swapTemps(gpus, dummy, N):
    # CPU implementation of PT swap
    t1 = time.time()

    for gpu in gpus:
        gpu.calcEnergies('main')
    es, Bs = map(np.concatenate, readGPUbufs(['E main', 'Bs'], gpus))
    ns = len(es)

    #Bs_orig = Bs.copy()
    #r1 = logaddexp.reduce(-Bs*es)/len(Bs)

    # swap consecutive replicas, where consecutive is in E order
    order = np.argsort(es)
    #order = arange(len(es))
    es = es[order]
    Bs = Bs[order]

    deEvn = es[ :-1:2] - es[1::2]
    deOdd = es[1:-1:2] - es[2::2]

    # views of even and odd elements
    eBs1, eBs2 = Bs[:-1:2], Bs[1::2]
    oBs1, oBs2 = Bs[1:-1:2], Bs[2::2]

    def swap(a, b, dE):
        ind_diff = np.where(a != b)[0]
        delta = dE[ind_diff]*(a[ind_diff] - b[ind_diff])
        ind_lz = np.where(delta < 0)[0]
        sind = np.ones(len(delta), dtype='bool')
        sind[ind_lz] = np.exp(delta[ind_lz]) > rand_f32(len(ind_lz))
        sind = ind_diff[sind]

        a[sind], b[sind] = b[sind], a[sind]

    for n in range(N):
        swap(eBs1, eBs2, deEvn)
        swap(oBs1, oBs2, deOdd)

    # get back to original order
    Bs[order] = Bs.copy()

    r = 0
    #r = sum(Bs != Bs_orig)/float(len(Bs))
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
        gpu.calcEnergies('main')
    bicounts, energies = readGPUbufs(['bicount', 'E main'], gpus)
    bicounts = sumarr(bicounts)
    bimarg_model = bicounts.astype('f4')/np.float32(np.sum(bicounts[0,:]))
    energies = np.concatenate(energies)

    if savedir:
        np.save(os.path.join(savedir, 'bimarg_{}'.format(step)), bimarg_model)
        np.save(os.path.join(savedir, 'energies_{}'.format(step)), energies)
    return energies, bimarg_model

def runMCMC(gpus, couplings, runName, param, log):
    nloop = param.equiltime
    trackequil = param.trackequil
    outdir = param.outdir
    # assumes small sequence buffer is already filled

    #get ready for MCMC
    for gpu in gpus:
        gpu.setBuf('J', couplings)

    #equilibration MCMC
    if nloop == 'auto':
        equil_dir = os.path.join(outdir, runName, 'equilibration')
        mkdir_p(equil_dir)

        loops = 16
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

            rstr = "Step {} <E>={:.2f}. ".format(step, np.mean(energies))

            if len(equil_e) >= 3:
                r1, p1 = spearmanr(equil_e[-1], equil_e[-2])
                r2, p2 = spearmanr(equil_e[-1], equil_e[-3])

                fmt = "({:.3f}, {:.2g}). ".format
                rstr += "r=cur:{} prev:{}".format(fmt(r1, p1), fmt(r2, p2))

                if p1 > 0.02 and p2 > 0.02:
                    log(rstr + "Equilibrated.")
                    break

            loops = loops*2
            log(rstr + "Continuing equilibration.")

        e_rho = [spearmanr(ei, equil_e[-1]) for ei in equil_e]

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
        e_rho = [spearmanr(ei, equil_e[-1]) for ei in equil_e]

    #process results
    for gpu in gpus:
        gpu.calcBicounts('main')
        gpu.calcEnergies('main')
    bicount, es = readGPUbufs(['bicount', 'E main'], gpus)
    bicount = sumarr(bicount)
    bimarg_model = bicount.astype(np.float32)/np.float32(np.sum(bicount[0,:]))
    sampledenergies = np.concatenate(es)

    return bimarg_model, bicount, sampledenergies, e_rho, None

def runMCMC_tempered(gpus, couplings, runName, param, log):
    nloop = param.equiltime
    trackequil = param.trackequil
    outdir = param.outdir
    # assumes small sequence buffer is already filled

    B0 = np.max(param.tempering)

    #get ready for MCMC
    for gpu in gpus:
        gpu.setBuf('J', couplings)

    #equilibration MCMC
    if nloop == 'auto':
        equil_dir = os.path.join(outdir, runName, 'equilibration')
        mkdir_p(equil_dir)

        loops = 16
        for i in range(loops):
            for gpu in gpus:
                gpu.runMCMC()
        step = loops

        equil_e = []
        while True:
            for i in range(loops):
                for gpu in gpus:
                    gpu.runMCMC()
                Bs,r = swapTemps(gpus, param.tempering, param.nswaps)

            step += loops
            energies, _ = track_main_bufs(gpus, equil_dir, step=step)
            equil_e.append(energies)
            np.save(os.path.join(outdir, runName,
                    'equilibration', 'Bs_{}'.format(step)), np.concatenate(Bs))

            if len(equil_e) >= 3:
                r1, p1 = spearmanr(equil_e[-1], equil_e[-2])
                r2, p2 = spearmanr(equil_e[-1], equil_e[-3])

                fmt = "({:.3f}, {:.2g})".format
                rstr = "Step {}, r=cur:{} prev:{}".format(
                        step, fmt(r1, p1), fmt(r2, p2))

                # Note that we are testing the correlation for *all* walkers,
                # no matter their temperature. In other words, we are waiting
                # for both the temperatures and energies to equilibrate - each
                # walker is expected to visit most temperatures during the
                # equilibration.
                if p1 > 0.02 and p2 > 0.02:
                    log(rstr + ". Equilibrated.")
                    break
            else:
                rstr = "Step {}".format(step)

            loops = loops*2
            log(rstr + ". Continuing equilibration.")

        e_rho = [spearmanr(ei, equil_e[-1]) for ei in equil_e]
    elif trackequil == 0:
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

            energies, _ = track_main_bufs(gpus, equil_dir, step=step)
            np.save(os.path.join(outdir, runName,
                    'equilibration', 'Bs_{}'.format(j)), np.concatenate(Bs))

            equil_e.append(energies)

        # track how well different walkers are equilibrated. Should go to 0
        e_rho = [spearmanr(ei, equil_e[-1]) for ei in equil_e]

    for B,gpu in zip(Bs,gpus):
        gpu.markSeqs(B == B0)
        gpu.clearLargeSeqs()
        gpu.storeMarkedSeqs() #save seqs from smallbuf to largebuf

    #process results
    for gpu in gpus:
        gpu.calcBicounts('large')
        gpu.calcEnergies('large')
    res = readGPUbufs(['bicount', 'E large'], gpus)
    bicount = sumarr(res[0])
    # assert sum(bicount, axis=1) are all equal here
    bimarg_model = bicount.astype(np.float32)/np.float32(np.sum(bicount[0,:]))
    sampledenergies = np.concatenate(res[1])

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
    log("(Re-centering gauge of couplings)")
    couplings = fieldlessGaugeEven(np.zeros((L,q)), couplings)[1]

    mkdir_p(os.path.join(outdir, runName))
    np.save(os.path.join(outdir, runName, 'J'), couplings)

    MCMC_func = runMCMC
    if param.tempering is not None:
        MCMC_func = runMCMC_tempered

    start_time = time.time()

    log("Equilibrating MCMC chains...")
    (bimarg_model,
     bicount,
     sampledenergies,
     e_rho,
     ptinfo) = MCMC_func(gpus, couplings, runName, param, log)

    seqs = readGPUbufs(['seq main'], gpus)[0]

    end_time = time.time()

    #get summary statistics and output them
    writeStatus(runName, bimarg_target, bicount, bimarg_model,
                couplings, seqs, sampledenergies,
                alpha, e_rho, ptinfo, outdir, log)
    log("Total MCMC running time: {:.1f} s".format(end_time - start_time))

    if param.tempering is not None:
        e, b = readGPUbufs(['E main', 'Bs'], gpus)
        np.save(os.path.join(outdir, runName, 'walker_Es'), np.concatenate(e))
        np.save(os.path.join(outdir, runName, 'walker_Bs'), np.concatenate(b))

    #compute new J using local newton updates (in-place on GPU)
    if param.newtonSteps != 1:
        newcouplings, bimarg_p = iterNewton(param, bimarg_model, gpus, log)
        np.save(os.path.join(outdir, runName, 'predictedBimarg'), bimarg_p)
        np.save(os.path.join(outdir, runName, 'predictedJ'), newcouplings)
    else:
        log("Performing single newton update step")
        newcouplings = singleNewton(bimarg_model, param.gamma0,
                                    param, gpus)

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
        couplings = preOpt(param, gpus, log)
    elif param.preequiltime:
        log("Pre-equilibration for {} steps...".format(param.preequiltime))
        log("(Re-centering gauge of couplings)")
        L, q = param.L, param.q
        couplings = fieldlessGaugeEven(np.zeros((L,q)), couplings)[1]
        if param.resetseqs:
            for gpu in gpus:
                gpu.fillSeqs(seedseq)
        for gpu in gpus:
            gpu.setBuf('J', couplings)
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
    if param.reg == 'X':
        for gpu in gpus:
            gpu.setBuf('Creg', param.regarg)
    if param.reg == 'Xself':
        for gpu in gpus:
            gpu.setBuf('Xlambdas', param.regarg)

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
            nseq = np.sum(s.shape[0] for s in seqs)
            rseq = None
            rseq_ind = randint(0, nseq)
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


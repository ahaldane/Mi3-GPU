# Copyright 2019 Allan Haldane.
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
import numpy as np
from scipy.stats import pearsonr, dirichlet, spearmanr
from numpy.random import randint
import pyopencl as cl
import sys, os, errno, glob, argparse, time
from utils.changeGauge import fieldlessGaugeEven
from utils.seqload import writeSeqs, loadSeqs
from utils import printsome, getLq, indep_bimarg, unimarg
from Mi3 import generateSequences, MPI, mkdir_p
import time

################################################################################
#Helper funcs

def printstats(name, jstep, bicount, bimarg_target, bimarg_model, couplings,
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
{name} J-Step {jstep: 6d}  Error: SSR: {ssr: 7.3f}  Ferr: {ferr: 7.5f}  X: {X}
{name} bimarg: {bimarg} ...
{name}      J: {couplings} ...
{name} min(E) = {lowE:.4f}     mean(E) = {meanE:.4f}     std(E) = {stdE:.4f}
{name} E Autocorr vs time: {rhos}""".format(
        name=name, jstep=jstep, ferr=ferr, ssr=ssr,
        X="{: 7.3f} ({: 7.3f})".format(np.sum(X), np.sum(Xo)),
        bimarg=printsome(bimarg_model),
        couplings=printsome(couplings), lowE=min(energies),
        meanE=np.mean(energies), stdE=np.std(energies), rhos=rhostr)

    if ptinfo != None:
        disp += "\n{} PT swap rate: {}".format(name, ptinfo[1])

    return disp

def writeStatus(name, Jstep, bimarg_target, bicount, bimarg_model, couplings,
                seqs, energies, alpha, e_rho, ptinfo, outdir, log):

    dispstr = printstats(name, Jstep, bicount, bimarg_target, bimarg_model,
                         couplings, energies, e_rho, ptinfo)
    with open(os.path.join(outdir, name, 'info.txt'), 'wt') as f:
        f.write(dispstr)

    #save current state to file
    np.savetxt(os.path.join(outdir, name, 'bicounts'), bicount, fmt='%d')
    np.save(os.path.join(outdir, name, 'bimarg'), bimarg_model)
    np.save(os.path.join(outdir, name, 'energies'), energies)
    writeSeqs(os.path.join(outdir, name, 'seqs'), seqs, alpha, zipf=True)
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

    gpus.setBuf('bi', bimarg)
    # note: updateJ should give same result on all GPUs
    # overwrites J front using bi back and J back
    if param.reg == 'l2z':
        lh, lJ = param.regarg
        gpus.updateJ_l2z(gamma, pc, lh, lJ)
    elif param.reg == 'X':
        gpus.updateJ_X(gamma, pc)
    elif param.reg == 'Xself':
        gpus.updateJ_Xself(gamma, pc)
    else:
        gpus.updateJ(gamma, pc)
    return gpus.head_gpu.getBuf('J')[0].read()

def getNeff(w):
    # This corresponds to an effective N for the weighted average of N
    # bernoulli trials, eg X = (w1 X1 + w2 X2 + ...)/sum(w) for which we find
    # var(X) = p(1-p)/Neff, with Neff = N in the unweighted case.
    return (np.sum(w)**2)/np.sum(w**2)

def NewtonStatus(n, trialJ, weights, bimarg_model, bimarg_target, log):
    SSR = np.sum((bimarg_model.flatten() - bimarg_target.flatten())**2)

    # not clear what best Neff measure is. sum(weights)?
    Neff = getNeff(weights)
    log("Predicted statistics after perturbing J:")
    log("     SSR: {:.4f}    Neff: {:.1f}    wspan: {:.3g}:{:.3g}".format(
         SSR, Neff, min(weights), max(weights)))
    log("  trialJ:", printsome(trialJ)), '...'
    log("  bimarg:", printsome(bimarg_model)), '...'
    log(" weights:", printsome(weights)), '...'

def iterNewton(param, bimarg_model, gpus, log):
    bimarg_target = param.bimarg
    gamma = param.gamma0
    newtonSteps = param.newtonSteps
    pc = param.pcdamping
    Nfrac = param.fracNeff
    N = param.nwalkers

    log("")
    log("Perturbation optimization for up to {} steps:".format(newtonSteps))
    if param.reg:
        log("Using regularization {}".format(param.reg))

    s = time.time()

    head_node = gpus.head_node

    seqbuf = 'main'
    wbufname = 'weights'
    ebufname = 'E main'
    if param.distribute_jstep != 'all':
        seqs = gpus.collect('seq main')

        if param.distribute_jstep == 'head_gpu':
            gpus = gpus.head_gpu
            log("Transferring sequences to gpu 0")
        if param.distribute_jstep == 'head_node':
            gpus = gpus.head_node
            log("Transferring sequences to node 0")

        gpus.setSeqs('large', [seqs])
        seqbuf = 'large'
        wbufname = 'weights large'
        ebufname = 'E large'

    gpus.calcEnergies(seqbuf)
    gpus.calcBicounts(seqbuf)
    gpus.bicounts_to_bimarg(seqbuf)
    gpus.merge_bimarg()

    if param.reg == 'l2z':
        updateJ = lambda: gpus.updateJ_l2z(gamma, pc, *param.regarg, Jbuf='dJ')
    #elif param.reg == 'l1z':
    #    updateJ = lambda: gpus.updateJ_l1z(gamma, pc, *param.regarg, Jbuf='dJ')
    elif param.reg == 'X':
        updateJ = lambda: gpus.updateJ_X(gamma, pc, Jbuf='dJ')
    elif param.reg == 'Xself':
        updateJ = lambda: gpus.updateJ_Xself(gamma, pc, Jbuf='dJ')
    else:
        updateJ = lambda: gpus.updateJ(gamma, pc, Jbuf='dJ')

    gpus.fillBuf('dJ', 0)

    # do coupling updates
    lastNeff = 2*N
    for i in range(newtonSteps):
        updateJ()
        if param.reg == 'l1z':
            gpus.reg_l1z(gamma, pc, *param.regarg)
        gpus.calcEnergies(seqbuf, 'dJ')

        # compute weights on CPU (since we want to subtract max for precision)
        dJ = gpus.readBufs(ebufname)
        mindJ = np.min([np.min(x) for x in dJ])
        weights = [np.exp(mindJ - x) for x in dJ]
        gpus.setBuf(wbufname, weights)

        gpus.weightedMarg(seqbuf)
        gpus.merge_bimarg()
    
        weights = np.concatenate(weights)
        Neff = getNeff(weights)
        if i%64 == 0 or abs(lastNeff - Neff)/N > 0.05 or Neff < Nfrac*N:
            log("J-step {: 5d}   Neff: {:.1f}   ({:.1f}% of {})".format(
                 i, Neff, Neff/N*100, N))
            lastNeff = Neff
        if Neff < Nfrac*N:
            log("Ending coupling updates because Neff/N < {:.2f}".format(Nfrac))
            break
    log("Performed {} coupling update steps".format(i))

    # print status
    bi, J, dJ = gpus.head_gpu.readBufs(['bi', 'J', 'dJ'])
    bimarg_model, trialJ = bi[0], J[0] + dJ[0]
    gpus.setBuf('J', trialJ)
    NewtonStatus(i, trialJ, weights, bimarg_model, bimarg_target, log)

    Neff = np.sum(weights)
    if not np.isfinite(Neff) or Neff == 0:
        raise Exception("Error: Divergence. Decrease gamma or increase "
                        "pc-damping")

    e = time.time()
    log("Total Newton-step running time: {:.1f} s".format(e-s))

    # dump profiling info if profiling is turned on
    gpus.logProfile()

    return i, trialJ, bimarg_model


################################################################################

#pre-optimization steps
def preOpt(param, gpus, log):
    J = param.couplings
    outdir = param.outdir
    alpha = param.alpha
    bimarg_target = param.bimarg

    log("Pre-Optimization:")

    # we assume that at this point the "main" sequence buffers are filled with
    # sequences
    gpus.setBuf('J', J)
    gpus.calcBicounts('main')
    gpus.calcEnergies('main')
    bicount, es, seqs = gpus.collect(['bicount', 'E main', 'seq main'])
    bimarg = bicount.astype(np.float32)/np.float32(np.sum(bicount[0,:]))

    mkdir_p(os.path.join(param.outdir, 'preopt'))
    writeStatus('preopt', 0, bimarg_target, bicount, bimarg,
                J, seqs, es, alpha, None, None, outdir, log)

    Jsteps, newJ = NewtonSteps('preopt', param, bimarg, gpus, log)

    return newJ, Jsteps

def rand_f32(size):
    """
    Generate random float32 values [0, 1), assuming little-endian IEEE binary32

    This is a speed optimization. In the IEEE binary32 floating point format,
    such values all have the same exponent and only vary in the mantissa. So,
    we only need to generate the 23 bit mantissa per uniform float32. It turns
    out in numpy it is fastest to generate 32 bits and then drop the first 9.
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

    # swap matching inds in a, b, with PT swap prob
    def swap(a, b, dE):
        ind_diff = np.where(a != b)[0]
        delta = dE[ind_diff]*(a[ind_diff] - b[ind_diff])
        ind_lz = np.where(delta < 0)[0]
        sind = np.ones(len(delta), dtype='bool')
        sind[ind_lz] = np.exp(delta[ind_lz]) > rand_f32(len(ind_lz))
        sind = ind_diff[sind]

        a[sind], b[sind] = b[sind], a[sind]

    # iterate the swaps, alternating left and right swaps
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

def track_main_bufs(param, gpus, savedir=None, step=None, saveseqs=False):
    gpus.calcBicounts('main')
    gpus.calcEnergies('main')

    bicounts, energies = gpus.collect(['bicount', 'E main'])
    bimarg_model = bicounts.astype('f4')/np.float32(np.sum(bicounts[0,:]))

    if saveseqs:
        seqs = gpus.collect('seq main')

    if savedir:
        np.save(os.path.join(savedir, 'bimarg_{}'.format(step)), bimarg_model)
        np.save(os.path.join(savedir, 'energies_{}'.format(step)), energies)

        if saveseqs:
            writeSeqs(os.path.join(savedir, 'seqs_{}'.format(step)), seqs, 
                      param.alpha, zipf=True)
    return energies, bimarg_model

def runMCMC(gpus, couplings, runName, param, log):
    nloop = param.equiltime
    trackequil = param.trackequil
    outdir = param.outdir
    # assumes small sequence buffer is already filled

    #get ready for MCMC
    gpus.setBuf('J', couplings)

    #equilibration MCMC
    if nloop == 'auto':
        equil_dir = os.path.join(outdir, runName, 'equilibration')
        mkdir_p(equil_dir)

        loops = 8
        for i in range(loops):
            gpus.runMCMC()
        step = loops

        equil_e = []
        while True:
            for i in range(loops):
                gpus.runMCMC()

            step += loops
            energies, _ = track_main_bufs(param, gpus, equil_dir, step)
            equil_e.append(energies)

            rstr = "Step {} <E>={:.2f}. ".format(step, np.mean(energies))

            if len(equil_e) >= 3:
                r1, p1 = spearmanr(equil_e[-1], equil_e[-2])
                r2, p2 = spearmanr(equil_e[-1], equil_e[-3])

                fmt = "({:.3f}, {:.2g}) ".format
                rstr += "r={} prev:{}".format(fmt(r1, p1), fmt(r2, p2))

                if p1 > 0.02 and p2 > 0.02 and step >= param.min_equil:
                    log(rstr + "Equilibrated.")
                    break

            log(rstr + "Continuing.")
            loops = loops*2

        e_rho = [spearmanr(ei, equil_e[-1]) for ei in equil_e]

    elif trackequil == 0:
        #keep nloop iterator on outside to avoid filling queue with only 1 gpu
        for i in range(nloop):
            gpus.runMCMC()

        step = nloop
        e_rho = None

    else:
        #note: sync necessary with trackequil (may slightly affect performance)
        equil_dir = os.path.join(outdir, runName, 'equilibration')
        mkdir_p(equil_dir)
        equil_e = []
        for j in range(nloop//trackequil):
            for i in range(trackequil):
                gpus.runMCMC()
            energies, _ = track_main_bufs(param, gpus, equil_dir, j*trackequil)
            equil_e.append(energies)

        step = nloop
        # track how well different walkers are equilibrated. Should go to 0
        e_rho = [spearmanr(ei, equil_e[-1]) for ei in equil_e]

    #process results
    gpus.calcBicounts('main')
    gpus.calcEnergies('main')
    bicount, es = gpus.collect(['bicount', 'E main'])
    bimarg_model = bicount/np.sum(bicount[0,:])

    gpus.logProfile()

    return bimarg_model, bicount, es, e_rho, None, step

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

        loops = 8
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
            energies, _ = track_main_bufs(param, gpus, equil_dir, step)
            equil_e.append(energies)
            np.save(os.path.join(outdir, runName,
                    'equilibration', 'Bs_{}'.format(step)), np.concatenate(Bs))

            if len(equil_e) >= 3:
                r1, p1 = spearmanr(equil_e[-1], equil_e[-2])
                r2, p2 = spearmanr(equil_e[-1], equil_e[-3])

                fmt = "({:.3f}, {:.2g})".format
                rstr = "Step {}, r={} prev:{}".format(
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
            log(rstr + ". Continuing.")

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
        for j in range(nloop//trackequil):
            for i in range(trackequil):
                for gpu in gpus:
                    gpu.runMCMC()
                Bs,r = swapTemps(gpus, param.tempering, param.nswaps)

            energies, _ = track_main_bufs(param, gpus, equil_dir, step)
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

    for gpu in gpus:
        gpu.logProfile()

    return bimarg_model, bicount, sampledenergies, e_rho, (Bs, r)

def NewtonSteps(runName, param, bimarg_model, gpus, log):
    outdir = param.outdir

    #compute new J using local newton updates (in-place on GPU)
    if param.newtonSteps != 1:
        Jsteps, newJ, bimarg_p = iterNewton(param, bimarg_model, gpus, log)
        np.save(os.path.join(outdir, runName, 'predictedBimarg'), bimarg_p)
        np.save(os.path.join(outdir, runName, 'perturbedJ'), newJ)
    else:
        log("Performing single newton update step")
        newJ = singleNewton(bimarg_model, param.gamma0, param, gpus)
        Jsteps = 1

    return Jsteps, newJ

def MCMCstep(runName, Jstep, couplings, param, gpus, log):
    outdir = param.outdir
    alpha, L, q = param.alpha, param.L, param.q
    bimarg_target = param.bimarg

    log("")
    log("Gradient Descent step {}".format(runName))
    log("---------------------------")
    log("Total J update step {}".format(Jstep))

    #re-distribute energy among couplings
    #(not really needed, but makes nicer output and might prevent
    # numerical inaccuracy, but also shifts all seq energies)
    log("(Re-zeroing gauge of couplings)")
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
     ptinfo,
     equilsteps) = MCMC_func(gpus, couplings, runName, param, log)

    end_time = time.time()

    #get summary statistics and output them
    seqs = gpus.collect('seq main')
    writeStatus(runName, Jstep, bimarg_target, bicount, bimarg_model,
                couplings, seqs, sampledenergies,
                alpha, e_rho, ptinfo, outdir, log)

    dt = end_time - start_time
    log("Total MCMC running time: {:.1f} s    ({:.3g} MC/s)".format(
        dt, equilsteps*param.nsteps*np.float64(gpus.nwalkers)/dt))

    if param.tempering is not None:
        e, b = gpus.collect(['E main', 'Bs'])
        np.save(os.path.join(outdir, runName, 'walker_Es'), e)
        np.save(os.path.join(outdir, runName, 'walker_Bs'), b)

    Jsteps, newJ = NewtonSteps(runName, param, bimarg_model, gpus, log)

    return Jstep + Jsteps, seqs, sampledenergies, newJ

def newtonMCMC(param, gpus, start_run, log):
    J = param.couplings

    # copy target bimarg to gpus
    gpus.setBuf('bi target', param.bimarg)

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

    # setup up regularization if needed
    if param.reg == 'X':
        gpus.setBuf('Creg', param.regarg)
    if param.reg == 'Xself':
        gpus.setBuf('Xlambdas', param.regarg)

    # pre-optimization
    Jsteps = 0
    if param.preopt:
        J, Jsteps = preOpt(param, gpus, log)
    else:
        log("No Pre-optimization")
    
    # do some setup for reseed options
    seqs = param.seqs
    seedseq = param.seedseq
    if seqs is not None and param.reseed == 'single_best':
        gpus.calcEnergies()
        es = gpus.collect('E main')

    # solve using newton-MCMC
    Jstep = Jsteps
    name_fmt = 'run_{{:0{}d}}'.format(int(np.ceil(np.log10(param.mcmcsteps))))
    for i in range(start_run, param.mcmcsteps):
        runname = name_fmt.format(i)

        # determine seed sequence, if using seed
        seed = None
        if seedseq is not None:
            seed = seedseq
            seedseq = None  # only use provided seed in first round
        elif param.reseed == 'single_indep':
            seed = generateSequences('independent', param.L, param.q, 1,
                                     param.bimarg, log)[0]
        elif param.reseed == 'single_random':
            #choose random seed from the final sequences from last round
            nseq = np.sum(s.shape[0] for s in seqs)
            seed = seqs[randint(0, nseq)]
        elif param.reseed == 'single_best':
            seed = seqs[np.argmin(es)]

        # fill sequence buffers (with seed or otherwise)
        mkdir_p(os.path.join(param.outdir, runname))
        if seed is not None:
            with open(os.path.join(param.outdir, runname, 'seedseq'),'wt') as f:
                f.write("".join(param.alpha[c] for c in seed))
            gpus.fillSeqs(seed)
        elif param.reseed == 'independent':
            indep_seqs = generateSequences('independent', param.L, param.q,
                                            gpus.nwalkers, param.bimarg, log)
            gpus.setSeqs('main', indep_seqs)
        elif param.reseed == 'msa':
            gpus.setSeqs('main', param.seedmsa)

        Jstep, seqs, es, J = MCMCstep(runname, Jstep, J, param, gpus, log)


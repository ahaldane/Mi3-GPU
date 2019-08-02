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
from mcmcGPU import MCMCGPU
import time

def sumarr(arrlist):
    #low memory usage (rather than sum(arrlist, axis=0))
    tot = arrlist[0].copy()
    for a in arrlist[1:]:
        np.add(tot, a, tot)
    return tot

def meanarr(arrlist):
    return sumarr(arrlist)/len(arrlist)

# optimization: skip reduction ops for inputs with only one element
def skip_single(f):
    def skipper(x):
        if len(x) == 1:
            return x[0]
        return f(x)
    return skipper

# object which manages a set of GPUs on a single node. Provides
# a basic interface for coordinating computations on a group of GPUs.

class GPU_node:
    def __init__(self, gpus):
        self.gpus = gpus
        self.head_node = self

    @property
    def head_gpu(self):
        return type(self)([self.gpus[0]])
    # note this returns a GPU_node with ngpus == 1.

    @property
    def nwalkers(self):
        return sum(g.nwalkers for g in self.gpus)

    @property
    def nseq(self):
        return {'main': [g.nseq['main'] for g in self.gpus],
                'large': [g.nseq.get('large', 0) for g in self.gpus]}

    @property
    def ngpus(self):
        return len(self.gpus)

    @property
    def gpu_list(self):
        return [("({}) ".format(g.gpunum) + g.device.name) for g in self.gpus]

    def initMCMC(self, nsteps, rng_offsets, rng_span):
        for gpu, offset in zip(self.gpus, rng_offsets):
            gpu.initMCMC(nsteps, offset, rng_span)

    def initLargeBufs(self, nseq):
        if isinstance(nseq, list):
            for n, gpu in zip(nseq, self.gpus):
                gpu.initLargeBufs(n)
        else:
            for gpu in self.gpus:
                gpu.initLargeBufs(nseq)

    def initJstep(self):
        for gpu in self.gpus:
            gpu.initJstep()

    def logProfile(self):
        for gpu in self.gpus:
            gpu.logProfile()

    def runMCMC(self):
        for gpu in self.gpus:
            gpu.runMCMC()

    def calcEnergies(self, seqbufname):
        for gpu in self.gpus:
            gpu.calcEnergies(seqbufname)

    def calcBicounts(self, seqbufname):
        for gpu in self.gpus:
            gpu.calcBicounts(seqbufname)

    def bicounts_to_bimarg(self, seqbufname='main'):
        for gpu in self.gpus:
            gpu.bicounts_to_bimarg(seqbufname)

    def updateJ(self, gamma, pc):
        for gpu in self.gpus:
            gpu.updateJ(gamma, pc)

    def updateJ_l2z(self, gamma, pc, lh, lJ):
        for gpu in self.gpus:
            gpu.updateJ_l2z(gamma, pc, ls, lJ)

    def updateJ_X(self, gamma, pc):
        for gpu in self.gpus:
            gpu.updateJ_X(gamma, pc)

    def updateJ_Xself(self, gamma, pc):
        for gpu in self.gpus:
            gpu.updateJ_Xself(gamma, pc)

    def calcWeights(self, seqbufname):
        for gpu in self.gpus:
            gpu.calcWeights(seqbufname)

    def weightedMarg(self, seqbufname):
        for gpu in self.gpus:
            gpu.weightedMarg(seqbufname)

    def renormalize_bimarg(self):
        for gpu in self.gpus:
            gpu.renormalize_bimarg()

    def getBuf(self, bufname):
        return [gpu.getBuf(bufname) for gpu in self.gpus]

    def readBufs(self, bufnames):
        if isinstance(bufnames, str):
            return [buf.read() for buf in self.getBuf(bufnames)]
        futures = [self.getBuf(bn) for bn in bufnames]
        return [[fut.read() for fut in buff] for buff in futures]

    groupfunc = {
        'weights': skip_single(np.concatenate),
        'E': skip_single(np.concatenate),
        'Bs': skip_single(np.concatenate),
        'bicount': skip_single(sumarr),
        'bi': skip_single(meanarr),
        'seq': skip_single(np.concatenate)}
    def collect(self, bufs):
        if isinstance(bufs, str):
            buftype = bufs.split()[0]
            func = self.groupfunc.get(buftype, lambda x: x)
            return func(self.readBufs(bufs))

        ret = []
        for bufname, arrs in zip(bufs, self.readBufs(bufs)):
            buftype = bufname.split()[0]
            func = self.groupfunc.get(buftype, lambda x: x)
            ret.append(func(arrs))

        return ret

    def setBuf(self, bufname, dat):
        if isinstance(dat, list):
            for gpu, buf in zip(self.gpus, dat):
                gpu.setBuf(bufname, buf)
        else:
            for gpu in self.gpus:
                gpu.setBuf(bufname, dat)

    def setSeqs(self, bufname, seqs, log=None):
        
        if isinstance(seqs, np.ndarray) or len(seqs) == 1:
            # split up seqs into parts for each gpu
            if not isinstance(seqs, np.ndarray):
                seqs = seqs[0]
            sizes = self.nseq[bufname]

            if len(seqs) != sum(sizes):
                raise Exception(("Expected {} total sequences, got {}").format(
                                 sum(sizes), len(seqs)))

            seqs = np.split(seqs, np.cumsum(sizes)[:-1])
        elif len(seqs) != self.ngpus:
            raise Exception(("Expected {} sequence bufs, got {}").format(
                             self.ngpus, len(seqs)))
        
        if log:
            log("Transferring {} seqs to gpu's {} seq buffer...".format(
                                         str([len(s) for s in seqs]), bufname))
        for n,(gpu,seq) in enumerate(zip(self.gpus, seqs)):
            gpu.setBuf('seq ' + bufname, seq)

    def fillSeqs(self, seq):
        for gpu in self.gpus:
            gpu.fillSeqs(seq)

    def storeSeqs(self, seqs=None):
        for gpu in self.gpus:
            gpu.storeSeqs(seqs)

    def clearLargeSeqs(self):
        for gpu in self.gpus:
            gpu.clearLargeSeqs()

    def reduce_node_bimarg(self):
        # each gpu has its own bimarg computed for its sequences. We want to
        # sum the bimarg to get the total bimarg, so need to share the bimarg
        # buffers across devices. Since gpus are in same context, they can
        # share buffers.

        # There are a few possible transfer strategies with different bus
        # usage/sync issues. The one below uses a divide by two strategy: In
        # the first round, second half of gpus send to first half, eg gpu 3 to
        # 1, 2 to 0, which do a sum. Then repeat on first half of gpus,
        # dividing gpus in half each round, then broadcast final result from
        # gpu 0.

        # make sure we are done writing to buffers on all devices
        # (since queue on one device does not wait for other devices)

        self.wait()

        rgpus = self.gpus
        while len(rgpus) > 1:
            h = (len(rgpus)-1)//2 + 1
            for even, odd in zip(rgpus[:h], rgpus[h:]):
                even.addBiBuffer('bi', odd.bufs['bi'])

            for even in rgpus[:h]:
                even.wait()

            rgpus = rgpus[:h]

    def merge_bimarg(self):
        if len(self.gpus) == 1:
            return

        self.reduce_node_bimarg()

        # sum above was not normalized (GPUs may contribute different weights)
        self.gpus[0].renormalize_bimarg()
        # don't start transfers until gpu0 is done. We need to be careful with
        # cross-gpu stuff because each gpu uses a different queue, so operations
        # across queues are not necessarily ordered.
        self.wait() 
    
        # take advantage of between-gpu tranfers if possible (no CPU transfer)
        bibuf = self.gpus[0].bufs['bi']
        for g in self.gpus[1:]:
            g.setBuf('bi', bibuf)

        # wait so gpu0 doesn't overwrite before other gpus are done copying
        self.wait() 

    def wait(self):
        for g in self.gpus:
            g.wait()

    def logProfile(self):
        for g in self.gpus:
            g.logProfile()

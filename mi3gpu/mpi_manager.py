# Copyright 2020 Allan Haldane.
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

import time, collections
from mpi4py import MPI
import pyopencl as cl
import numpy as np

from mi3gpu.node_manager import GPU_node, sumarr, meanarr

t0 = time.time()

mpi_comm = MPI.COMM_WORLD

# This file contains three classes which inherit the basic
# interface of the GPU_node class for managing I3-GPU computations on a set
# of GPUs.
#
# MPI_GPU_node_controller inherits this interface for managing a set of nodes,
# each of which has a set of GPUs. In this way MPI_GPU_node_controller can be
# used on the manager node as a drop-in replacement for a GPU_node instance, if
# using multiple nodes.
#
# MPI_GPU_node and MPI_worker together through MPI to provide a GPU_node
# interface on the manager node, in which the actual computation is performed on
# a worker node.

# This runs on the manager node
class MPI_multinode_controller(GPU_node):
    # this stores a set of nodes, which each have a set of gpus.

    def __init__(self, node_managers):
        self.nodes = node_managers
        self.head_node = node_managers[0]

        # also call this gpus so we can inherit most functions
        self.gpus = node_managers

    @property
    def head_gpu(self):
        return self.head_node.head_gpu

    @property
    def ngpus(self):
        return sum(n.ngpus for n in self.nodes)

    @property
    def gpu_list(self):
        return [name + ' on node {}'.format(n)
                for n, node in enumerate(self.nodes) for name in node.gpu_list]

    def _zip_gpus(self, lst):
        pos = 0
        for n in self.nodes:
            yield n, lst[pos:pos + n.ngpus]
            pos += n.ngpus

    def _initMCMC_rng(self, nsteps, rng_offsets, rng_span):
        for node, offset in self._zip_gpus(rng_offsets):
            node._initMCMC_rng(nsteps, offset, rng_span)

    def initLargeBufs(self, nseq):
        if isinstance(nseq, list):
            for node, n in self._zip_gpus(nseq):
                node.initLargeBufs(n)
        else:
            for node in self.nodes:
                node.initLargeBufs(nseq)

    def getBuf(self, bufname):
        return sum((node.getBuf(bufname) for node in self.nodes), [])

    def collect(self, bufs):
        # each worker has already used groupfunc to collect from GPUs on that
        # node. Now we use groupfunc to collect those results across nodes.
        arrs = (n.collect(bufs) for n in self.nodes)
        #XXX not anync....

        if isinstance(bufs, str):
            buftype = bufs.split()[0]
            func = self.groupfunc.get(buftype, lambda x: x)
            return func(list(arrs))

        ret = []
        for bufname, a in zip(bufs, zip(*arrs)):
            buftype = bufname.split()[0]
            func = self.groupfunc.get(buftype, lambda x: x)
            ret.append(func(a))
        return ret

    def setBuf(self, bufname, dat):
        if isinstance(dat, list):
            for node, bufs in self._zip_gpus(dat):
                node.setBuf(bufname, bufs)
        else:
            for node in self.nodes:
                node.setBuf(bufname, dat)

    def setSeqs(self, bufname, seqs, log=None):
        if isinstance(seqs, np.ndarray) or len(seqs) == 1:
            # split up seqs into parts for each gpu
            if not isinstance(seqs, np.ndarray):
                seqs = seqs[0]
            sizes = self.nseq[bufname]

            if len(seqs) == sum(sizes):
                raise Exception(("Expected {} total sequences, got {}").format(
                                 sum(sizes), len(seqs)))
            seqs = np.split(seqs, np.cumsum(sizes)[:-1])
        elif len(seqs) != self.ngpus:
            raise Exception(("Expected {} sequence bufs, got {}").format(
                             self.ngpus, len(seqs)))

        if log:
            log("Transferring {} seqs to gpu's {} seq buffer...".format(
                                         str([len(s) for s in seqs]), bufname))

        for n,(node,node_seq) in enumerate(self._zip_gpus(seqs)):
            node.setSeqs(bufname, node_seq, None)

    def reduce_node_bimarg(self):
        for n in self.nodes:
            n.reduce_node_bimarg()

    def merge_bimarg(self):
        self.reduce_node_bimarg()

        head_bi = self.head_node.gpus[0].getBuf('bi')
        bibufs = [head_bi] + [n.get_head_bi() for n in self.nodes[1:]]
        # final sum done on CPU.
        bi = sumarr([b.read() for b in bibufs])
        bi = bi/np.sum(bi, axis=1, keepdims=True) # renormalize

        self.setBuf('bi', bi)

    def wait(self):
        for n in self.nodes:
            n.wait()

    def runMCMC(self):
        for n,gpu in enumerate(self.gpus):
            gpu.runMCMC()

class MPI_comm_Mixin:
    def __init__(self):
        self.waitlist = collections.deque()
        self._msg_counts = {}

    def check_waitlist(self):
        while len(self.waitlist) > 0 and self.waitlist[0].Test():
            self.waitlist.popleft()

    def _debug(self, cmd, tag, val=None):
        if 0:
            n = self._msg_counts.get((cmd, tag), 0)
            print(">>>>>>{} {: 8.3f}    {} {}/{}:{} <-> {}   {} ".format(
                  '>>>'*self.self_rank, time.time() - t0,
                  self.self_rank, cmd, tag, n, self.other_rank, val or ''))
            self._msg_counts[(cmd, tag)] = n+1

    def isend(self, val, tag=0):
        self._debug('isend', tag, repr(val)[:40])
        rq = mpi_comm.isend(val, dest=self.other_rank, tag=tag)
        self.waitlist.append(rq)
        self.check_waitlist()

    def Isend_raw(self, dat, tag=0):
        # this version does not send shape/dtype info, it is callers
        # responsibility to do so
        self._debug('Isend', tag, val='arr: {} {}'.format(dat.shape, dat.dtype))
        rq = mpi_comm.Isend(dat, dest=self.other_rank, tag=tag)
        self.waitlist.append(rq)
        self.check_waitlist()

    def Isend(self, dat, tag=0):
        self.isend((dat.shape, dat.dtype), tag=tag)
        self.Isend_raw(dat, tag=tag)
        # we use Isend for performance to avoid a pickle-copy. But in
        # python3.8 pickle protocol-5 will be no-copy, so can replace with
        # an isend.

    def recv(self, tag=0):
        #self._debug('recv', tag)
        #return mpi_comm.recv(source=self.other_rank, tag=tag)
        ret = mpi_comm.recv(source=self.other_rank, tag=tag)
        self._debug('recv', tag, repr(ret)[:40])
        return ret

    def Recv(self, tag=0):
        shape, dtype = self.recv(tag=tag)
        dat = np.empty(shape, dtype)
        self._debug('Recv', tag, val='arr: {} {}'.format(shape, dtype))
        mpi_comm.Recv(dat, source=self.other_rank, tag=tag)
        return dat

    def Irecv(self, tag=0):
        shape, dtype = self.recv(tag=tag)
        dat = np.empty(shape, dtype)
        self._debug('IRecv', tag, val='arr: {} {}'.format(shape, dtype))
        return FutureBuf_MPI(dat, self.other_rank, tag)


class FutureBuf_MPI:
    def __init__(self, buffer, src_rank, tag):
        self.rq = mpi_comm.Irecv(buffer, source=src_rank, tag=tag)

        self.buffer = buffer

        self.shape = buffer.shape
        self.dtype = buffer.dtype

    def read(self):
        self.rq.wait()
        return self.buffer


# this runs on the manager node
class MPI_GPU_node(GPU_node, MPI_comm_Mixin):
    def __init__(self, rank, ngpus):
        MPI_comm_Mixin.__init__(self)

        self.self_rank = 0
        self.other_rank = rank

        # we store node param on manager side for efficiency here
        self._ngpus = self.recv()
        self._nwalkers = self.recv()
        self._nseq = self.recv()

        # as a sanity check, confirm number of gpus with worker node
        assert(ngpus == self._ngpus)

    @property
    def ngpus(self):
        return self._ngpus

    @property
    def gpu_list(self):
        self.isend('get_gpu_list')
        return self.recv()

    @property
    def nwalkers(self):
        return self._nwalkers

    @property
    def nseq(self):
        return self._nseq

    @property
    def head_gpu(self):
        return self

    def _initMCMC_rng(self, nsteps, rng_offsets, rng_span):
        self.isend('_initMCMC_rng')
        self.isend((nsteps, rng_offsets, rng_span))

    def initLargeBufs(self, nseq):
        self.isend('initLargeBufs')
        self.isend(nseq)

    def initJstep(self):
        self.isend('initJstep')

    def runMCMC(self):
        self.isend('runMCMC')

    def calcEnergies(self, seqbufname, Jbufname='J'):
        self.isend('calcEnergies')
        self.isend(seqbufname)
        self.isend(Jbufname)

    def calcBicounts(self, seqbufname):
        self.isend('calcBicounts')
        self.isend(seqbufname)

    def bicounts_to_bimarg(self, seqbufname='main'):
        self.isend('bicounts_to_bimarg')
        self.isend(seqbufname)

    def updateJ(self, gamma, pc, Jbuf='dJ'):
        self.isend('updateJ')
        self.isend((gamma, pc, Jbuf))

    def reg_l1z(self, gamma, pc, lJ):
        self.isend('reg_l1z')
        self.isend((gamma, pc, lJ))

    def reg_l2z(self, gamma, pc, lJ):
        self.isend('reg_l2z')
        self.isend((gamma, pc, lJ))

    def reg_X(self, gamma, pc):
        self.isend('reg_X')
        self.isend((gamma, pc))

    def calcWeights(self, seqbufname):
        self.isend('calcWeights')
        self.isend(seqbufname)

    def weightedMarg(self, seqbufname):
        self.isend('weightedMarg')
        self.isend(seqbufname)

    def renormalize_bimarg(self):
        self.isend('renormalize_bimarg')

    # Note on MPI message ordering with tags:
    #
    # We use tag 0 for all messages except GPU buffer transfers in getBuf,
    # where tags 1 to ngpus+1 are used instead. Thus, the tag represents a
    # transfer from a gpu. This is to allow asynchronous coordination of the
    # GPU->CPU transfer with the worker->manager transfer in getBufs. The
    # problem is that we cannot emit an mpi_send from the worker CPU until the
    # CPU receives the GPU buffer, which conflicts with the "non-overtaking"
    # order guarantee of MPI in case we want to emit other mpi_sends in the
    # meantime.
    #
    # The solution is to use tags, as the "non-overlapping" guarantee only
    # appies to messages with the same source/tag combination, allowing
    # reordering of different tags. We still have to make sure MPI sends
    # from a gpu tag are in the right order. For this we depend on
    # the fact that our cl_enqueue_copy calls preserve order too, so that
    # their completion happens in the same order as initiation, so that
    # we can use mpi_send in the cl-callback and get the right order.
    # (While technically we use an out-of-order CL queue, we actually never
    # make any out of order calls in practice. If we do in the future, just
    # make sure the GPU copy calls are ordered).

    def getBuf(self, bufname):
        self.isend('getBuf')
        self.isend(bufname)

        bufs = []
        for n in range(self._ngpus):
            bufs.append(self.Irecv(tag=n+1))
        return bufs

    def collect(self, bufs):
        # this only gets the 'collected' data from the worker, and passes
        # it back to the multinode_controller for futher collection.
        self.isend('collect')
        self.isend(bufs)

        if isinstance(bufs, str):
            return self.Recv()
        return [self.Recv() for n in range(len(bufs))]

    def setBuf(self, bufname, dat):
        self.isend('setBuf')
        self.isend(bufname)
        self.isend(isinstance(dat, list))

        if isinstance(dat, list):
            assert(len(dat) == self.ngpus)
            for n, buf in enumerate(dat):
                self.Isend(buf)
        else:
            self.Isend(dat)

    def fillBuf(self, bufname, val):
        self.isend('fillBuf')
        self.isend(bufname)
        self.isend(val)

    def setSeqs(self, bufname, seqs, log=None):
        # note, unlike node_manager.seqSeqs, here seqs must be a list of len == ngpus
        self.isend('seqSeqs')
        self.isend(bufname)
        assert(len(seqs) == self.ngpus)
        for buf in seqs:
            self.Isend(buf)

    def fillSeqs(self, seq):
        self.isend('fillSeqs')
        #print(repr(seq), type(seq))
        self.Isend(seq)

    def storeSeqs(self, seqs=None):
        self.isend('storeSeqs')
        self.isend(seqs is None)
        if seqs is not None:
            self.Isend(seqs)

    def clearLargeSeqs(self):
        self.isend('clearLargeSeqs')

    def reduce_node_bimarg(self):
        self.isend('reduce_node_bimarg')

    def get_head_bi(self):
        self.isend('get_head_bi')
        return self.Irecv()

    def merge_bimarg(self):
        # this is implemented on manager's node_controller
        # (when using MPI we want to do this across nodes, not a single node)
        raise NotImplementedError

    def wait(self):
        self.isend('wait')

    def logProfile(self):
        self.isend('logProfile')


# this runs on a worker node
class MPI_worker(GPU_node, MPI_comm_Mixin):
    def __init__(self, rank, gpus):
        MPI_comm_Mixin.__init__(self)

        self.gpus = gpus

        self.self_rank = rank
        self.other_rank = 0

        # send back basic properties to manager node
        self.isend(self.ngpus)
        self.isend(self.nwalkers)
        self.isend(self.nseq)

    def listen(self):
        while True:
            #Note: mpi releases the GIL in blocking calls, so CL callbacks run
            command = self.recv()

            if command == 'exit':
                break

            getattr(self, command)()

    def get_gpu_list(self):
        self.isend(self.gpu_list)

    def _initMCMC_rng(self):
        args = self.recv()
        super()._initMCMC_rng(*args)

    def initLargeBufs(self):
        nseq = self.recv()
        super().initLargeBufs(nseq)

    def calcEnergies(self):
        seqbufname = self.recv()
        Jbufname = self.recv()
        super().calcEnergies(seqbufname, Jbufname)

    def calcBicounts(self):
        seqbufname = self.recv()
        super().calcBicounts(seqbufname)

    def bicounts_to_bimarg(self):
        seqbufname = self.recv()
        super().bicounts_to_bimarg(seqbufname)

    def updateJ(self):
        args = self.recv()
        super().updateJ(*args)

    def reg_l1z(self):
        args = self.recv()
        super().reg_l1z(*args)

    def reg_l2z(self):
        args = self.recv()
        super().reg_l2z(*args)

    def reg_X(self):
        args = self.recv()
        super().reg_X(*args)

    def calcWeights(self):
        seqbufname = self.recv()
        super().calcWeights(seqbufname)

    def weightedMarg(self):
        seqbufname = self.recv()
        super().weightedMarg(seqbufname)

    def renormalize_bimarg(self):
        super().renormalize_bimarg()

    def setBuf(self):
        bufname = self.recv()
        is_list = self.recv()

        # Use blocking Recv for now. Do we want non-blocking? Probably, since
        # CL enqueue_copy is non-blocking, but then again it shouldn't matter
        # for now since we do all CL ops in order anyway.
        if is_list:
            dat = [self.Recv() for n in range(self.ngpus)]
        else:
            dat = self.Recv()
        super().setBuf(bufname, dat)

    def fillBuf(self, bufname, val):
        bufname = self.recv()
        val = self.recv()
        super().fillBuf(bufname, val)

    def getBuf(self):
        def make_cl_callback(b, n):
            def callback(s):
                # handle errors in s?
                self.Isend_raw(b.read(), tag=n)
            return callback

        bufname = self.recv()
        bufs = super().getBuf(bufname)
        for n, b in enumerate(bufs):
            # we send shape/dtype info before callback, since it gives
            # manager more time to allocate a buffer (faster)
            self.isend((b.shape, b.dtype), tag=n+1)
            b.event.set_callback(cl.command_execution_status.COMPLETE,
                                 make_cl_callback(b, n+1))

    def readBufs(self):
        # this is implemented on manager node in MPI_GPU_node
        raise NotImplementedError

    def super_readBufs(self, bufnames):
        # WARNING: This is not like the others, it is not an MPI callback.
        # This actually gets the buffers from the gpus.
        s = super()
        if isinstance(bufnames, str):
            return [buf.read() for buf in s.getBuf(bufnames)]
        futures = [s.getBuf(bn) for bn in bufnames]
        return [[fut.read() for fut in buff] for buff in futures]

    def collect(self):
        # we group with groupfunc on the worker before MPI transfer,
        # so that we transfer less data
        bufs = self.recv()

        if isinstance(bufs, str):
            buftype = bufs.split()[0]
            func = self.groupfunc.get(buftype, lambda x: x)
            self.Isend(func(self.super_readBufs(bufs)))
            return

        for bufname, arrs in zip(bufs, self.super_readBufs(bufs)):
            buftype = bufname.split()[0]
            func = self.groupfunc.get(buftype, lambda x: x)
            self.Isend(func(arrs))

    def setSeqs(self, bufname, seqs, log=None):
        bufname = self.recv()
        seqs = [self.Recv(n+1) for n in range(self.ngpus)]
        for gpu,seq in zip(self.gpus, seqs):
            gpu.setBuf('seq ' + bufname, seq)

    def fillSeqs(self):
        seq = self.Recv()
        super().fillSeqs(seq)

    def storeSeqs(self):
        have_buf = self.recv()
        seqs = None
        if have_buf:
            seqs = self.Recv()

        for gpu in self.gpus:
            gpu.storeSeqs(seqs)

    def merge_bimarg(self):
        # this is implemented on manager's node_controller
        raise NotImplementedError

    def get_head_bi(self):
        buf = self.gpus[0].getBuf('bi')
        self.isend((buf.shape, buf.dtype))
        buf.event.set_callback(cl.command_execution_status.COMPLETE,
                               lambda s: self.Isend_raw(buf.read()))

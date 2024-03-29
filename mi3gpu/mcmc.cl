//Copyright 2020 Allan Haldane.
//
//This file is part of Mi3-GPU.
//
//Mi3-GPU is free software: you can redistribute it and/or modify
//it under the terms of the GNU General Public License as published by
//the Free Software Foundation, version 3 of the License.
//
//Mi3-GPU is distributed in the hope that it will be useful,
//but WITHOUT ANY WARRANTY; without even the implied warranty of
//MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//GNU General Public License for more details.
//
//You should have received a copy of the GNU General Public License
//along with Mi3-GPU.  If not, see <http://www.gnu.org/licenses/>.
//
//Contact: allan.haldane _AT_ gmail.com
#include <mwc64x/cl/mwc64x/mwc64xvec2_rng.cl>
#include <mwc64x/cl/mwc64x/mwc64x_rng.cl>

#ifdef cl_khr_byte_addressable_store
#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
#else
#error "The cl_khr_byte_addresssable_store extension is required!!"
#endif

#pragma OPENCL EXTENSION cl_khr_fp64: enable

float uniformMap(uint i) {
    return (i>>8)*0x1.0p-24f; //converts a 32 bit integer to a float [0,1)
}

#define NCOUPLE ((L*(L-1)*q*q)/2)
#define SWORDS ((L-1)/4+1)

#define getbyte(mem, n) (((uchar*)(mem))[n])
#define setbyte(mem, n, val) {(((uchar*)(mem))[n]) = val;}

// WGSIZE must be pow of 2
#define WGMASK(x) (x&(~(WGSIZE-1)))  // round down to multiple of WGSIZE

//expands couplings stored in a (nPair x q*q) form to an (L*L x q*q) form
__kernel //to be called with group size q*q, with nPair groups
void unpackfV(__global float *v,
              __global float *vp) {
    uint li = get_local_id(0);
    uint gi = get_group_id(0);

    //figure out which i,j pair we are
    uint i = 0;
    uint j = L-1;
    while (j <= gi) {
        i++;
        j += L-1-i;
    }
    j = gi + L - j; //careful with underflow!

    //note: Does not fill in diagonal terms (i == j)!

    __local float lv[q*q];
    lv[li] = v[gi*q*q + li];
    barrier(CLK_LOCAL_MEM_FENCE);
    vp[q*q*(L*i+j) + li] = lv[li];
    vp[q*q*(i+L*j) + li] = lv[q*(li%q) + li/q];
}

// copies sequences in smallbuf to end of largebuf
__kernel
void storeSeqs(__global uint *smallbuf,
               __global uint *largebuf,
                        uint  nlargebuf,
                        uint  offset) {
    uint w;
    uint nseqs = get_global_size(0);
    uint n = get_global_id(0);

    for (w = 0; w < SWORDS; w++) {
        largebuf[w*nlargebuf + offset + n] = smallbuf[w*nseqs + n];
    }
}

// copies some sequences from large buf to small buf
__kernel
void restoreSeqs(__global uint *smallbuf,
                 __global uint *largebuf,
                          uint  nlargebuf,
                          uint  offset) {
    uint w, n;
    uint nseqs = get_global_size(0);

    for (w = 0; w < SWORDS; w++) {
        for (n = get_local_id(0); n < nseqs; n += get_local_size(0)) {
            smallbuf[w*nseqs + n] = largebuf[w*nlargebuf + offset + n];
        }
    }
}

// copies fixed positions from a sequence in the small buffer to those
// positions in the large buffer. Call with large-buffer-nseq work units.
__kernel
void copySubseq(__global uint *smallbuf,
                __global uint *largebuf,
                         uint  nsmallbuf,
                         uint  seqnum,
               __constant uchar *fixedpos) {
    uint sbs, sbl;
    uint pos, lastmod=0xffffffff;

    for (pos = 0; pos < L; pos++) {
        if (fixedpos[pos]) {
            if (pos/4 != lastmod/4) {
                sbl = largebuf[(pos/4)*get_global_size(0) + get_global_id(0)];
                sbs = smallbuf[(pos/4)*nsmallbuf + seqnum];
            }
            setbyte(&sbl, pos%4, getbyte(&sbs, pos%4));
            lastmod = pos;
        }
        if ( (((pos+1)%4 == 0) || (pos+1 == L)) && lastmod/4 == pos/4) {
            largebuf[(pos/4)*get_global_size(0) + get_global_id(0)] = sbl;
        }
    }
}

// reformats sequence memory from 4-byte rows to 1-byte rows. Ie, normally
// sequences are stored in SWORDS rows (nseq columns) as:
//         row1:   a1 a2 a3 a4 b1 b2 b3 b4 c1 c2 c3 c4 ...
//         row2:   a5 a6 a7 a8 b5 b6 b7 b8 c5 c6 c7 c8 ...
// where a1 is byte 1 of sequence a. This reformats to L rows
// and ((nseq-1)//4+1 cols) as:
//         row1:   a1 b1 c1 d1 e1 f1 ...
//         row2:   a2 b2 c2 d2 e2 f2 ...
//  (which is the transpose of seq array in CPU)
__kernel
void unpackseqs1T(__global uint *buf4,
                           uint  buf4len, //nseq uints rows
                  __global uint *buf1,
                           uint  buf1len) //nseq/4 uints
{
    uint i4 = get_group_id(0);
    uint li = get_local_id(0);
    __local uint scratch[256];

    for (uint n = 0; n < buf4len; n += 256) { //assumes buf4len multiple of 256
        // read in 256 values from row i4
        scratch[li] = buf4[i4*buf4len + n + li];
        barrier(CLK_LOCAL_MEM_FENCE);

        // repartition bytes in each group of 4 wu. Avoid bank conflicts.
        uint tmp = 0, s;
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            s = scratch[4*(li/4) + ((li+i)%4)];
            setbyte(&tmp, (li+i)%4, getbyte(&s, li%4));
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // reorganize the 256 uints into 4 grous of 64
        scratch[64*(li%4) + (li/4)] = tmp;
        barrier(CLK_LOCAL_MEM_FENCE);

        // write back 64 values to each of 4 rows of seqs
        uint outrow = 4*i4 + li/64;
        if (outrow < L) { //account for trailing padding in buf4
            buf1[buf1len*outrow + (n/4) + (li%64)] = scratch[li];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}

__kernel
void gen_indep(__global uint *buf4,
                        uint  buf4len, //nseq uints rows
 __global mwc64xvec2_state_t *rngstates,
              __global float *pbuf)
{
    uint gi = get_global_id(0);
    uint li = get_local_id(0);
    __local float cprobs[q-1];
    float prefetch_cprob;
    if (li < q-1) {
        prefetch_cprob = pbuf[li];
    }

    uint2 rng;
    mwc64xvec2_state_t rstate = rngstates[gi];

    // pbuf is a (q-1)*L buffer of cumulprobs, where each row of size q-1
    // represents cumulative probabilities of amino acids (excluding the last
    // element which must be 1)
    // group size of WGSIZE assumed to be larger than q.

    // loop over positions
    uint sbn = 0;
    int pos;
    for(pos = 0; pos < L; pos++) {
        // load next prob values
        if (li < q-1) {
            cprobs[li] = prefetch_cprob;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (pos < L-1) {
            if (li < q-1) {
                prefetch_cprob = pbuf[(pos+1)*(q-1) + li];
            }
        }

        // generate random number from our uint2 stream
        float r;
        if (pos%2 == 0) {
            rng = MWC64XVEC2_NextUint2(&rstate);
            r = uniformMap(rng.x);
        }
        else {
            r = uniformMap(rng.y);
        }

        // binary search for mutres
        uint bot = 0, mid;
        #pragma unroll
        for (uint span = q; span > 1; span -= mid) {
            mid = span / 2;
            if (r >= cprobs[bot + mid - 1]) {
                bot += mid;
            }
        }

        // bot specifies mutres now 
        setbyte(&sbn, pos%4, bot);

        if ((pos+1)%4 == 0) {
            buf4[(pos/4)*buf4len + gi] = sbn;
        }
    }
    // write out final uint if needed
    if (pos%4 != 0) {
        // zero out padding bytes (may be unnecessary..)
        for (int l = pos%4; l < 4; l++) {
            setbyte(&sbn, l, 0);
        }
        buf4[(pos/4)*buf4len + gi] = sbn;
    }

    rngstates[gi] = rstate;
}

// ****************************** Energy computation **************************

// this function expects J in "packed" form, with nPair*q*q elements
inline float getEnergiesf(__global float *J,
                          __global  uint *seqmem,
                                    uint  buflen,
                          __local  float *lJ) {
// This function is complicated by optimizations for the GPU.
// For clarity, here is equivalent but clearer (pseudo)code:
//
//    float energy = 0;
//    uint pair = 0;
//    for (n = 0; n < L-1; n++) {
//        for (m = n+1; m < L; m++) {
//           energy += J[pair,seq[n],seq[m]];
//           pair++;
//        }
//    }
//
// Idea of optimized J loads: We use a 2*WGSIZE local J buffer, and require
// that WGSIZE >= q*q. At start, we fill the buffer and also prefetch the next
// (third) WGSIZE J values to register. The local buffer can be thought of as
// two WGSIZE halves, A and B, from which we access q*q coulplings (a row) at a
// time. We iterate in chunks of q*q, but stop after the q*q chunk goes past
// the local mem B end. Then store the next prefetch values to the A half, use
// modulo indexing to access the local mem as if A followed B, and begin
// prefetch of the next WGSIZE Js. Continue iterating modulo chunks of q*q
// until we would hang past the end of A, then replace the B half with latest
// prefetch. Continue alternating the A and the B prefetches/overwrites.
//
// Note: J buffer must be padded with an extra 3*WGSIZE elements long,
// since we prefetch 3 groups of WGSIZE initially, and sometimes load an
// extra WGSIZE at the end of the J buffer.

    // load 2*WGSIZE worth of couplings
    uint Jmem_offset = 0;
    lJ[get_local_id(0)] = J[Jmem_offset + get_local_id(0)];
    Jmem_offset += WGSIZE;
    lJ[get_local_id(0) + WGSIZE] = J[Jmem_offset + get_local_id(0)];
    Jmem_offset += WGSIZE;
    barrier(CLK_LOCAL_MEM_FENCE);

    // prefetch next WGSIZE couplings
    float Jprefetch = J[Jmem_offset + get_local_id(0)];
    Jmem_offset += WGSIZE;

    uint lJ_offset = 0;

    float energy = 0;
    float rem = 0;

    uint n, sbn;
    for (n = 0; n < L-1; n++) {
        if (n%4 == 0) {
            sbn = seqmem[(n/4)*buflen + get_global_id(0)];
        }

        uint seqn = getbyte(&sbn, n%4);

        uint m, sbm = sbn;
        for (m = n+1; m < L; m++) {
            if (m%4 == 0) {
                sbm = seqmem[(m/4)*buflen + get_global_id(0)];
            }

            uint seqm = getbyte(&sbm, m%4);

            // Kahan summation for extra precision (should be essentially
            // no performance hit since hidden by memory latency of J loads)
            float y = lJ[(lJ_offset + q*seqn + seqm)%(2*WGSIZE)] - rem;
            float t = energy + y;
            rem = (t - energy) - y;
            energy = t;

            // load more couplings and advance lJ_offset if necessary
            if (lJ_offset + q*q >= WGMASK(lJ_offset) + WGSIZE) {
                // store prefetched values in their place
                barrier(CLK_LOCAL_MEM_FENCE);
                lJ[WGMASK(lJ_offset) + get_local_id(0)] = Jprefetch;
                barrier(CLK_LOCAL_MEM_FENCE);

                // start next prefetch
                Jprefetch = J[Jmem_offset + get_local_id(0)];
                Jmem_offset += WGSIZE;
            }

            lJ_offset = (lJ_offset + q*q)%(2*WGSIZE);
        }
    }
    return energy;
}

__kernel
void getEnergies(__global float *J,
                 __global uint *seqmem,
                          uint  buflen,
                 __global float *energies) {
    __local float lJ[2*WGSIZE];
    energies[get_global_id(0)] = getEnergiesf(J, seqmem, buflen, lJ);
}

// ****************************** Metropolis sampler **************************

__kernel
void initRNG2(__global mwc64xvec2_state_t *rngstates,
                       ulong offset,
                       ulong nsamples) {
    mwc64xvec2_state_t rstate;
    MWC64XVEC2_SeedStreams(&rstate, offset, nsamples);
    rngstates[get_global_id(0)] = rstate;
}

// This function is the bottleneck of the entire MCMC analysis.
// It is IO bound by the sequence loads and J loads.
inline float DeltaEnergy(__local float *lJ, __global float *J,
                          global uint *seqmem, uint nseqs,
                          uint pos, uint seqp, uchar mutres) {
    uint Jmem_offset = WGMASK(pos*L*q*q);
    uint lJ_offset = pos*L*q*q - Jmem_offset;;

    // load 2*WGSIZE worth of couplings
    lJ[get_local_id(0)] = J[Jmem_offset + get_local_id(0)];
    Jmem_offset += WGSIZE;
    lJ[get_local_id(0) + WGSIZE] = J[Jmem_offset + get_local_id(0)];
    barrier(CLK_LOCAL_MEM_FENCE);

    // prefetch next WGSIZE couplings
    Jmem_offset += WGSIZE;
    float Jprefetch = J[Jmem_offset + get_local_id(0)];

    float dE = 0;

    uint m, sbm;
    uint sbm_prefetch = seqmem[get_global_id(0)];
    for (m = 0; m < L; m++) {
        //loop through seq, changing energy by changed coupling with pos

        // load sequence data
        if (m%4 == 0) {
            sbm = sbm_prefetch;
            if (m+4 < L) {
                sbm_prefetch = seqmem[((m+4)/4)*nseqs + get_global_id(0)];
            }
        }

        if (m != pos) {
            uint seqm = getbyte(&sbm, m%4);
            dE += (lJ[(lJ_offset + q*mutres + seqm)%(2*WGSIZE)] -
                   lJ[(lJ_offset + q*seqp   + seqm)%(2*WGSIZE)]);
        }

        // load couplings
        if (lJ_offset + q*q >= WGMASK(lJ_offset) + WGSIZE) {
            // store prefetched values in their place
            barrier(CLK_LOCAL_MEM_FENCE);
            lJ[WGMASK(lJ_offset) + get_local_id(0)] = Jprefetch;
            barrier(CLK_LOCAL_MEM_FENCE);

            // start next prefetch
            Jmem_offset += WGSIZE;
            Jprefetch = J[Jmem_offset + get_local_id(0)];
        }

        lJ_offset = (lJ_offset + q*q)%(2*WGSIZE);
    }

    return dE;
}

__kernel
void metropolis(__global float *J,
                __global mwc64xvec2_state_t *rngstates,
                         uint position_offset,
                __global uint *position_list,
                         uint nsteps, // must be multiple of L
                __global float *energies, //ony used to measure fp error
                __global float *betas,
                __global uint *seqmem) {

    uint nseqs = get_global_size(0);
    mwc64xvec2_state_t rstate = rngstates[get_global_id(0)];

    //set up local mem
    __local float lJ[2*WGSIZE];

#ifdef TEMPERING
    float B = betas[get_global_id(0)];
#else
    const float B = BETA;
#endif

    uint i;
    for (i = 0; i < nsteps; i++) {
        uint pos = position_list[i + position_offset];
        uint2 rng = MWC64XVEC2_NextUint2(&rstate);
        rng.x = rng.x%q;         // small error here if MAX_INT%q != 0
        #define mutres  (rng.x)  // of order q/MAX_INT in marginals
        uint sbn = seqmem[(pos/4)*nseqs + get_global_id(0)];
        uint seqp = getbyte(&sbn, pos%4);

        float dE = DeltaEnergy(lJ, J, seqmem, nseqs, pos, seqp, mutres);

        //apply MC criterion and possibly update
        if (exp(-B*dE) > uniformMap(rng.y)) {
            setbyte(&sbn, pos%4, mutres);
            seqmem[(pos/4)*nseqs + get_global_id(0)] = sbn;
        }

        #undef mutres
    }

    rngstates[get_global_id(0)] = rstate;
}

// ****************************** Histogram Code **************************

// Note: This could be updated to use the faster algorithm in
// weightedMarg.
__kernel
void countBivariate(__global uint *bicount,
                             uint  nseq,
                    __global uint *seqmem,
                             uint  buflen,
                    __local  uint *hist) {
    uint li = get_local_id(0);
    uint gi = get_group_id(0);
    uint nhist = get_local_size(0);
    uint i,j,n,m;

    //figure out which i,j pair we are
    i = 0;
    j = L-1;
    while (j <= gi) {
        i++;
        j += L-1-i;
    }
    j = gi + L - j; //careful with underflow!

    for (n = 0; n < q*q; n++) {
        hist[nhist*n + li] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    uint tmp;
    // loop through all sequences
    for (n = li; n < nseq; n += nhist) {
        tmp = seqmem[(i/4)*buflen + n];
        uint seqi = ((uchar*)&tmp)[i%4];
        tmp = seqmem[(j/4)*buflen + n];
        uint seqj = ((uchar*)&tmp)[j%4];
        hist[nhist*(q*seqi + seqj) + li]++;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    //merge histograms
    for (n = 0; n < q*q; n++) {
        m = nhist;
        while (m > 1) {
            uint odd = m%2;
            m = (m+1)>>1; //div by 2 rounded up
            if (li < m - odd) {
                hist[nhist*n + li] = hist[nhist*n + li] + hist[nhist*n + li +m];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }

    for (n = li; n < q*q; n += nhist) { //only loops once if nhist > q*q
        bicount[gi*q*q + n] = hist[nhist*n];
    }
}

__kernel
void bicounts_to_bimarg(__global uint *bicount,
                        __global float *bimarg,
                                 uint  nseq) {
    uint n = get_global_id(0);
    if (n >= NCOUPLE) {
        return;
    }

    bimarg[n] = ((float)bicount[n])/((float)nseq);
}

__kernel //sums a vector. Call with 1 group of size VSIZE, must be power of two
void sumFloats(__global float *data,
               __global float *output,
                        uint  len,
               __local  float *sums) {
    uint li = get_local_id(0);
    uint vsize = get_local_size(0);
    uint n;

    // accumulate through array
    sums[li] = 0;
    for (n = li; n < len; n += vsize) {
        sums[li] += data[n];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    //reduce accumulated sums (req power of two unlike elsewhere in this file)
    for (n = vsize/2; n > 0; n >>= 1) {
        if (li < n) {
            sums[li] = sums[li] + sums[li + n];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    // assumes li is even. Need to add last element if odd
    if (li == 0) {
        *output = sums[0];
    }
}

__kernel //sums a vector. Call with 1 group of size VSIZE, must be power of two
void minFloats(__global float *data,
               __global float *output,
                         uint  len,
               __local  float *local_min) {
    uint li = get_local_id(0);
    uint vsize = get_local_size(0);
    uint n;

    // accumulate through array
    local_min[li] = 0;
    for (n = li; n < len; n += vsize) {
        local_min[li] = fmin(data[n], local_min[li]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    //reduce (req power of two unlike elsewhere in this file)
    for (n = vsize/2; n > 0; n >>= 1) {
        if (li < n) {
            local_min[li] = fmin(local_min[li], local_min[li + n]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    // assumes li is even. Need to add last element if odd
    if (li == 0) {
        *output = local_min[0];
    }
}

__kernel // Call with 1 group of size VSIZE, must be power of two
void weight_stats(__global float *data,
                  __global float *out3,
                  __local  float *local_sum,
                  __local  float *local_sum2,
                            uint  len) {
    uint li = get_local_id(0);
    const uint vsize = get_local_size(0);
    uint n;

    // accumulate
    local_sum[li] = 0;
    local_sum2[li] = 0;
    for (n = li; n < len; n += vsize) {
        float val = data[n];
        local_sum[li] += val;
        local_sum2[li] += val*val;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    //reduce (req power of two unlike elsewhere in this file)
    for (n = vsize/2; n > 0; n >>= 1) {
        if (li < n) {
            local_sum[li] = local_sum[li] + local_sum[li + n];
            local_sum2[li] = local_sum2[li] + local_sum2[li + n];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    // assumes li is even. Need to add last element if odd
    if (li < 2) {
        float tmp;
        switch(li) {
            case 0: tmp = local_sum[0]; break;
            case 1: tmp = local_sum2[0]; break;
        }
        out3[li] = tmp;
    }
}

__kernel
void dE_to_weights(         float ref_E,
                             uint buflen,
                   __global float *dE,
                   __global float *weights) {
    if (get_global_id(0) >= buflen) {
        return;
    }
    weights[get_global_id(0)] = exp(-dE[get_global_id(0)]+ref_E);
}

__kernel
void fixed_beta_weights(         float ref_E,
                                  uint buflen,
                        __global float *energies,
                        __global float *weights) {
    if (get_global_id(0) >= buflen) {
        return;
    }
    weights[get_global_id(0)] = exp((BETA-1)*(energies[get_global_id(0)]-ref_E));
}

// Idea: Have that NHIST, and HISTWS (work-size) are powers of 2, with
// HISTWS >= NHIST. Then in each loop over n below, we load memory as:
//      <----HISTWS---->
// si   xxxxxxxxxxxxxxxx  (remember these are packed uints, so expand
// sj   xxxxxxxxxxxxxxxx   4x to match w)
//  w   xxxxxxxxxxxxxxxx xxxxxxxxxxxxxxxx xxxxxxxxxxxxxxxx xxxxxxxxxxxxxxxx
//      <-NHIST->
//
// In each loop, load si, sj, and the 1st w segment. Then loop over all windows
// of NHIST inside HISTWS. Then load the next w, and loop k again, 4 times.
// Also, use fact that nseq is a multiple of 512, so need HISTWS <= 512.
__kernel //call with group size = 32, for nPair groups
void weightedMarg(__global float *bimarg_new,
                  __global float *weights,
                           uint nseq,
                  __global uint *seqmem,
                           uint  buflen) {
    uint li = get_local_id(0);
    uint gi = get_group_id(0);
    __local float hist[q*q*NHIST];
    __local uint sid[HISTWS], sjd[HISTWS];
    __local float w[HISTWS];
    uint n, m, i, j;

    //figure out which i,j pair we are
    i = 0;
    j = L-1;
    while (j <= gi) {
        i++;
        j += L-1-i;
    }
    j = gi + L - j; //careful with underflow!

    for (n = li; n < NHIST*q*q; n += HISTWS) {
        hist[n] = 0;
    }

    //loop through all sequences
    for (n = li; n < nseq/4; n += HISTWS) {
        sid[li] = seqmem[i*buflen + n];
        sjd[li] = seqmem[j*buflen + n];
        #pragma unroll
        for (uint k = 0; k < 4; k++) {
            w[li] = weights[HISTWS*(4*(n/HISTWS) + k) + li];
            barrier(CLK_LOCAL_MEM_FENCE);
            if (li < NHIST) {
                for (uint l = li; l < HISTWS; l += NHIST) {
                    uint si = sid[(HISTWS*k + l)/4];
                    uint sj = sjd[(HISTWS*k + l)/4];
                    uint bin = q*((uint)getbyte(&si, l%4)) + getbyte(&sj, l%4);
                    hist[NHIST*bin + li] += w[l];
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }

    //merge histograms. Every nhist/2 wu does a reduce over nhist elements
    uint x = li%(NHIST/2);
    for (n = li/(NHIST/2); n < q*q; n += HISTWS/(NHIST/2)) {
        // since NHIST is pow of two we can use simpler reduction code (no odd)
        for (m = NHIST/2; m > 0; m >>= 1) {
            if (x < m) {
                hist[NHIST*n + x] = hist[NHIST*n + x] + hist[NHIST*n + x + m];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }

    for (n = li; n < q*q; n += HISTWS) {
        bimarg_new[gi*q*q + n] = hist[NHIST*n];
    }
}

__kernel
void addFloatBufs(__global float *dst, __global float *src, int buflen) {
    uint n = get_global_id(0);
    if (n >= buflen) {
        return;
    }
    dst[n] += src[n];
}

// expects to be called with work-group size of q*q
// Local scratch memory must be provided:
// sums is q*q elements
float sumqq(float v, uint li, __local float *scratch) {
    uint m;

    scratch[li] = v;
    barrier(CLK_LOCAL_MEM_FENCE);

    // reduction loop which accounts for odd vector sizes
    m = q*q;
    while (m > 1) {
        uint odd = m%2;
        m = (m+1)>>1; //div by 2 rounded up
        if (li < m - odd) {
            scratch[li] = scratch[li] + scratch[li + m];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    return scratch[0];
    // Note: caller's responsibility to barrier before touching scratch!
}

// expects to be called with work-group size of q*q
__kernel
void renormalize_bimarg(__global float *bimarg) {
    __local float scratch[q*q];
    float bim = bimarg[get_global_id(0)];
    bimarg[get_global_id(0)] = bim/sumqq(bim, get_local_id(0), scratch);
}

// expects to be called with work-group size of q*q
// Local scratch memory must be provided:
// scratch is q*q elements, hi and hj are q elements.
float zeroGauge(float J, uint li, __local float *scratch,
                __local float *hi, __local float *hj) {
    uint m;

    // next couple lines are essentially a transform to the "zero" gauge

    //add up rows
    scratch[li] = J;
    barrier(CLK_LOCAL_MEM_FENCE);
    m = q;
    while (m > 1) {
        uint odd = m%2;
        m = (m+1)>>1; //div by 2 rounded up
        if (li%q < m - odd) {
            scratch[li] = scratch[li] + scratch[li + m];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (li < q) {
        hi[li] = scratch[q*li]/q;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    //add up columns
    scratch[q*(li%q) + li/q] = J;
    barrier(CLK_LOCAL_MEM_FENCE);
    m = q;
    while (m > 1) {
        uint odd = m%2;
        m = (m+1)>>1; //div by 2 rounded up
        if (li%q < m - odd) {
            scratch[li] = scratch[li] + scratch[li + m];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (li < q) {
        hj[li] = scratch[q*li]/q;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    //get total sum
    m = q;
    while (m > 1) {
        uint odd = m%2;
        m = (m+1)>>1; //div by 2 rounded up
        if (li < m - odd) {
            scratch[q*li] = scratch[q*li] + scratch[q*(li + m)];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    float mean = scratch[0]/q/q;
    barrier(CLK_LOCAL_MEM_FENCE);

    if (li < q) {
        hi[li] = hi[li] - mean;
        hj[li] = hj[li] - mean;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    return J - (hi[li/q] + hj[li%q]) - mean;
}

float fbnorm(float J, uint li, __local float *sums) {
    return sqrt(sumqq(J*J, li, sums));
}


__kernel
void updatedJ(__global float *bimarg_target,
              __global float *bimarg,
                       float gamma,
                       float pc,
              __global float *Ji,
              __global float *Jo) {
    uint n = get_global_id(0);

    if (n >= NCOUPLE) {
        return;
    }

    Jo[n] = Ji[n] - gamma*(bimarg_target[n] - bimarg[n])/(bimarg[n] + pc);
}

__kernel
void reg_l1z(__global float *bimarg,
                      float gamma,
                      float pc,
                      float lJ,
             __global float *J,
             __global float *dJ) {
    uint li = get_local_id(0);
    uint gi = get_group_id(0);
    uint n = gi*q*q + li;

    __local float hi[q], hj[q];
    __local float scratch[q*q];

    float Jp = dJ[n];
    float J0 = zeroGauge(J[n] + Jp, li, scratch, hi, hj);
    float R = -lJ*sign(J0)*gamma/(bimarg[n] + pc);

    // to reduce numerical fluctuations, if the regularization step
    // would change the sign of J0, instead set J0 to 0.
    if (sign(J0) != sign(J0 + R)){
        Jp = Jp - J0;
    }
    else {
        Jp = Jp + R;
    }
    dJ[n] = Jp;
}

__kernel
void reg_l2z(__global float *bimarg,
                      float gamma,
                      float pc,
                      float lJ,
             __global float *J,
             __global float *dJ) {
    uint li = get_local_id(0);
    uint gi = get_group_id(0);
    uint n = gi*q*q + li;

    __local float hi[q], hj[q];
    __local float scratch[q*q];

    float Jp = dJ[n];
    float J0 = zeroGauge(J[n] + Jp, li, scratch, hi, hj);
    float R = -lJ*J0*gamma/(bimarg[n] + pc);

    // to reduce numerical fluctuations, if the regularization step
    // would change the sign of J0, instead set J0 to 0.
    if (sign(J0) != sign(J0 + R)){
        Jp = Jp - J0;
    }
    else {
        Jp = Jp + R;
    }

    dJ[n] = Jp;
}

__kernel
void reg_SCADJ(__global float *bimarg,
                       float gamma,
                       float pc,
                       float lJ,
                       float a,
              __global float *J,
              __global float *dJ) {
    uint li = get_local_id(0);
    uint gi = get_group_id(0);
    uint n = gi*q*q + li;

    __local float hi[q], hj[q];
    __local float scratch[q*q];

    float Jp = dJ[n];
    float J0 = zeroGauge(J[n] + Jp, li, scratch, hi, hj);
    float R = 0;
    if (fabs(J0) < lJ) {
        // to avoid gamma-dependent oscillations around 0, set J0 exactly
        // to 0 if it would change sign, by appropriate clip on lJ.
        R = min(lJ, fabs(J0)/gamma);
    }
    else if (fabs(J0) < a*lJ) {
        R = (a*lJ - fabs(J0))/(a-1);
    }
    // account for step size and pseudocount damping in derivatives
    R *= gamma*sign(J0)/(bimarg[n] + pc);

    dJ[n] = Jp - R;
}

void getUnimarg(float fij, __local float *fi, __local float *fj, uint li, 
                __local float *scratch) {
    uint m;

    // add up bimarg rows
    scratch[li] = fij;
    barrier(CLK_LOCAL_MEM_FENCE);
    m = q;
    while (m > 1) {
        uint odd = m%2;
        m = (m+1)>>1; //div by 2 rounded up
        if (li%q < m - odd) {
            scratch[li] = scratch[li] + scratch[li + m];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (li < q) {
        fi[li] = scratch[q*li];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    //add up bimarg columns
    scratch[q*(li%q) + li/q] = fij;
    barrier(CLK_LOCAL_MEM_FENCE);
    m = q;
    while (m > 1) {
        uint odd = m%2;
        m = (m+1)>>1; //div by 2 rounded up
        if (li%q < m - odd) {
            scratch[li] = scratch[li] + scratch[li + m];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (li < q) {
        fj[li] = scratch[q*li];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
}

__kernel
void reg_Xij(__global float *bimarg,
           __global float *lambdas,
                    float gamma,
                    float pc,
           __global float *J,
           __global float *dJ) {
    uint li = get_local_id(0);
    uint gi = get_group_id(0);
    uint n = gi*q*q + li;

    __local float fi[q], fj[q];
    __local float scratch[q*q];

    float Jp = dJ[n];

    // XXX this derivative is missing a second term, hard to compute
    float fij = bimarg[n];
    getUnimarg(fij, fi, fj, li, scratch);
    float C = fij - fi[li/q]*fj[li%q];
    float X = sumqq((J[n] + Jp)*C, li, scratch);
    barrier(CLK_LOCAL_MEM_FENCE); // barrier for scratch usage
    float Xnorm = sumqq(C*C/(fij + pc), li, scratch);
    float lX = lambdas[gi];
    lX = min(lX, fabs(X)/(Xnorm*gamma));

    float R = lX*C*sign(X);
    dJ[n] = Jp - gamma*R/(fij+pc);
}

__kernel
void reg_X(__global float *bimarg,
                      float lX,
                      float gamma,
                      float pc,
             __global float *J,
             __global float *dJ) {
    uint li = get_local_id(0);
    uint gi = get_group_id(0);
    uint n = gi*q*q + li;

    __local float fi[q], fj[q];
    __local float scratch[q*q];

    float Jp = dJ[n];

    float fij = bimarg[n];
    getUnimarg(fij, fi, fj, li, scratch);
    float C = fij - fi[li/q]*fj[li%q];
    float X = sumqq((J[n] + Jp)*C, li, scratch);
    barrier(CLK_LOCAL_MEM_FENCE); // barrier for scratch usage
    float Xnorm = sumqq(C*C/(fij + pc), li, scratch);
    lX = min(lX, fabs(X)/(Xnorm*gamma));

    dJ[n] = Jp - gamma*lX*C*sign(X)/(fij+pc);
}

float Xijab(float J, float fij, __local float *fi, __local float *fj, uint li, 
            __local float *scratch) {
    uchar a, b, c, d;
    c = li%q;
    d = li/q;

    scratch[li] = J;
    barrier(CLK_LOCAL_MEM_FENCE);

    float Xijab;

    for (a = 0; a < q; a++) {
        for (b = 0; b < q; b++) {
            Xijab += scratch[a*q + b]*(fi[a] - (a == c))*(fj[b] - (b == d));
        }
    }
    return Xijab;
}

float getXijab(float J, __local float *fi, __local float *fj,
                uint li, __local float *scratch) {
    uchar a, b, c, d;
    c = li/q;
    d = li%q;

    scratch[li] = J;
    barrier(CLK_LOCAL_MEM_FENCE);

    float Xijab = 0;
    for (a = 0; a < q; a++) {
        for (b = 0; b < q; b++) {
            Xijab += scratch[a*q + b]*(fi[a] - (a == c))*(fj[b] - (b == d));
        }
    }
    return Xijab;
}


// SCAD is defined by a range r, and range multipler a:
//                R                               R'
//   |X| < r      r|X|                            r 
//   |X| < a*r    (2ar|X| - |X|^2 - r^2)/2(a-1)   (ar*s[X]-X)/(a-1)
//   |X| >= a*r   (1+a)r^2/2                      0
//
// below we also multiply by a scaling factor s such that D = s(1+a)r^2/2 so
// that D is the regularization cost for large |X| specified by user.
__kernel
void reg_SCADX(__global float *bimarg,
                       float gamma,
                       float pc,
                       float s,
                       float r,
                       float a,
              __global float *J,
              __global float *dJ) {
    uint li = get_local_id(0);
    uint gi = get_group_id(0);
    uint n = gi*q*q + li;

    __local float fi[q], fj[q];
    __local float scratch[q*q];

    float Jp = dJ[n];
    float Jt = J[n] + Jp;

    float fij = bimarg[n];
    getUnimarg(fij, fi, fj, li, scratch);
    float C = fij - fi[li/q]*fj[li%q];
    float X = sumqq((Jt)*C, li, scratch);
    barrier(CLK_LOCAL_MEM_FENCE); // barrier for scratch usage
    float Xnorm = sumqq(C*C/(fij + pc), li, scratch);
    barrier(CLK_LOCAL_MEM_FENCE); // barrier for scratch usage
    float Xijab = getXijab(Jt, fi, fj, li, scratch);

    float R = 0;
    if (fabs(X) < r) {
        // to avoid gamma-dependent oscillations around 0, set X exactly
        // to 0 if it would change sign, by appropriate clip on r.
        r = min(r, fabs(X)/(Xnorm*gamma*s));
        R = r*sign(X);
    }
    else if (fabs(X) < a*r) {
        // assume gamma is small enough X can't change sign if we are more 
        // than lX away from 0
        R = (a*r*sign(X) - X)/(a-1);
    }
    R *= gamma*s*(C + fij*(X - Xijab))/(fij + pc);

    dJ[n] = Jp - R;
}

__kernel
void reg_expX(__global float *bimarg,
                       float gamma,
                       float pc,
                       float lam,
              __global float *J,
              __global float *dJ) {
    uint li = get_local_id(0);
    uint gi = get_group_id(0);
    uint n = gi*q*q + li;

    __local float fi[q], fj[q];
    __local float scratch[q*q];

    float Jp = dJ[n];
    float Jt = J[n] + Jp;

    float fij = bimarg[n];
    getUnimarg(fij, fi, fj, li, scratch);
    float C = fij - fi[li/q]*fj[li%q];
    float X = sumqq((Jt)*C, li, scratch);
    barrier(CLK_LOCAL_MEM_FENCE); // barrier for scratch usage
    float Xijab = getXijab(Jt, fi, fj, li, scratch);

    float R = sign(X)*exp(-fabs(X)/lam)*(C + fij*(X - Xijab));

    dJ[n] = Jp - gamma*R/(fij + pc);
}

__kernel
void reg_ddE(__global float *bimarg,
                      float  gamma,
                      float  pc,
                      float  lambda,
             __global float *J,
             __global float *dJ) {
    uint li = get_local_id(0);
    uint gi = get_group_id(0);
    uint n = gi*q*q + li;

    __local float lJ[q*q];
    lJ[li] = J[n] + dJ[n];
    barrier(CLK_LOCAL_MEM_FENCE);

    #define a (li/q)
    #define b (li%q)
    #define JJ(a,b) lJ[q*(a) + (b)]

    float dR = 0;
    for (int g = 1; g < q; g++) {
        float jr = JJ(a, b) - JJ((a+g)%q, b);
        for (int d = 1; d < q; d++) {
            float jc = -JJ(a, (b+d)%q) + JJ((a+g)%q, (b+d)%q);
            float ddE = jr + jc;
            dR += sign(ddE);
            // set to 0 if abs(ddE) < gamma*lambda/f?
        }
    }
    dR = dR/((q-1)*(q-1)); // scale so irrelevant chars have no effect

    dJ[n] = dJ[n] - lambda*dR*gamma/(bimarg[n]+pc);

    #undef a
    #undef b
    #undef JJ
}



__kernel
void reg_SCADddE(__global float *bimarg,
                      float  gamma,
                      float  pc,
                      float  lambda,
                      float  r,
             __global float *J,
             __global float *dJ) {
    uint li = get_local_id(0);
    uint gi = get_group_id(0);
    uint n = gi*q*q + li;

    __local float lJ[q*q];
    lJ[li] = J[n] + dJ[n];
    barrier(CLK_LOCAL_MEM_FENCE);

    #define a (li/q)
    #define b (li%q)
    #define JJ(a,b) lJ[q*(a) + (b)]
    #define scale 4

    float dR = 0;
    for (int g = 1; g < q; g++) {
        float jr = JJ(a, b) - JJ((a+g)%q, b);
        for (int d = 1; d < q; d++) {
            float jc = -JJ(a, (b+d)%q) + JJ((a+g)%q, (b+d)%q);
            float ddE = jr + jc;

            float R = 0;
            float absD = fabs(ddE);
            if (absD < r) {
                R = 1;
            }
            else if (absD < scale*r) {
                R = (scale*r - absD)/(scale*r - r);
            }
            dR += lambda*sign(ddE)*R;
            // set to 0 if abs(ddE) < gamma*lambda/f?
        }
    }
    dR = dR/((q-1)*(q-1)); // scale so irrelevant chars have no effect
    // (since adding an extra q adds and extra row/col to double-sum above)
    // is this term properly incorporated into SCAD? I think so because
    // the dR ends up eqalling lambda for small AddE adter we do the sum
    // and division, in case of a single coupling.

    dJ[n] = dJ[n] - gamma*dR/(bimarg[n]+pc);

    #undef a
    #undef b
    #undef JJ
}

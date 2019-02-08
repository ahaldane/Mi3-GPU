//Copyright 2018 Allan Haldane.
//
//This file is part of IvoGPU.
//
//IvoGPU is free software: you can redistribute it and/or modify
//it under the terms of the GNU General Public License as published by
//the Free Software Foundation, version 3 of the License.
//
//IvoGPU is distributed in the hope that it will be useful,
//but WITHOUT ANY WARRANTY; without even the implied warranty of
//MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//GNU General Public License for more details.
//
//You should have received a copy of the GNU General Public License
//along with IvoGPU.  If not, see <http://www.gnu.org/licenses/>.
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

float uniformMap(uint i){
    return (i>>8)*0x1.0p-24f; //converts a 32 bit integer to a float [0,1)
}

#define NCOUPLE ((L*(L-1)*q*q)/2)
#define SWORDS ((L-1)/4+1)

#define getbyte(mem, n) (((uchar*)(mem))[n])
#define setbyte(mem, n, val) {(((uchar*)(mem))[n]) = val;}

//expands couplings stored in a (nPair x q*q) form to an (L*L x q*q) form
__kernel //to be called with group size q*q, with nPair groups
void unpackfV(__global float *v,
            __global float *vp){
    uint li = get_local_id(0);
    uint gi = get_group_id(0);

    //figure out which i,j pair we are
    uint i = 0;
    uint j = L-1;
    while(j <= gi){
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
                        uint  offset){
    uint w;
    uint nseqs = get_global_size(0);
    uint n = get_global_id(0);

    for(w = 0; w < SWORDS; w++){
        largebuf[w*nlargebuf + offset + n] = smallbuf[w*nseqs + n];
    }
}

// copies marked sequences in smallbuf to end of largebuf
__kernel
void storeMarkedSeqs(__global uint *smallbuf,
                     __global uint *largebuf,
                              uint  nlarge,
                              uint  offset,
                     __global uint *inds){
    // copies seqs from small seq buffer to large seq buffer
    // according to indices in "inds".

    uint w;
    uint nseqs = get_global_size(0);
    uint n = get_global_id(0);

    for(w = 0; w < SWORDS; w++){
        if(inds[n] != -1){
            largebuf[w*nlarge + offset + inds[n]] = smallbuf[w*nseqs + n];
        }
    }
}

// copies some sequences from large buf to small buf
__kernel
void restoreSeqs(__global uint *smallbuf,
                 __global uint *largebuf,
                          uint  nlargebuf,
                          uint  offset){
    uint w, n;
    uint nseqs = get_global_size(0);

    for(w = 0; w < SWORDS; w++){
        for(n = get_local_id(0); n < nseqs; n += get_local_size(0)){
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
               __constant uchar *fixedpos){
    uint sbs, sbl;
    uint pos, lastmod=0xffffffff;

    for(pos = 0; pos < L; pos++){
        if(fixedpos[pos]){
            if(pos/4 != lastmod/4){
                sbl = largebuf[(pos/4)*get_global_size(0) + get_global_id(0)];
                sbs = smallbuf[(pos/4)*nsmallbuf + seqnum];
            }
            setbyte(&sbl, pos%4, getbyte(&sbs, pos%4));
            lastmod = pos;
        }
        if( (((pos+1)%4 == 0) || (pos+1 == L)) && lastmod/4 == pos/4){
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
void unpackseqs1(__global uint *buf4,
                        uint  buf4len, //nseq uints rows
               __global uint *buf1,
                        uint  buf1len) //nseq/4 uints
{
    uint i4 = get_group_id(0);
    uint li = get_local_id(0);
    __local uint scratch[256];

    for(uint n = 0; n < buf4len; n += 256){ //assumes buf4len multiple of 256
        // read in 256 values from row i4
        scratch[li] = buf4[i4*buf4len + n + li];
        barrier(CLK_LOCAL_MEM_FENCE);

        // repartition bytes in each group of 4 wu. Avoid bank conflicts.
        uint tmp = 0, s;
        #pragma unroll
        for(int i = 0; i < 4; i++){
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

// ****************************** Energy computation **************************

// this function expects J in "packed" form, with nPair*q*q elements
inline float getEnergiesf(__global float *J,
                          __global uint *seqmem,
                                   uint  buflen,
                         __local float *lcouplings){
    // This function is complicated by optimizations for the GPU.
    // For clarity, here is equivalent but clearer (pseudo)code:
    //
    //float energy = 0;
    //uint pair = 0;
    //for(n = 0; n < L-1; n++){
    //    for(m = n+1; m < L; m++){
    //       energy += J[pair,seq[n],seq[m]];
    //       pair++;
    //    }
    //}

    uint li = get_local_id(0);

    uint seqm, seqn, seqp;
    uint cn, cm;
    uint sbn, sbm;
    uint n,m,k,pair;
    float energy = 0;
    pair = 0;
    n = 0;
    while(n < L-1){
        uint sbn = seqmem[(n/4)*buflen + get_global_id(0)];
        #pragma unroll //probably ignored
        for(cn = n%4; cn < 4 && n < L-1; cn++, n++){
            seqn = getbyte(&sbn, cn);
            m = n+1;
            while(m < L){
                uint sbm = seqmem[(m/4)*buflen + get_global_id(0)];

                #pragma unroll
                for(cm = m%4; cm < 4 && m < L; cm++, m++){
                    for(k = li; k < q*q; k += get_local_size(0)){
                        lcouplings[k] = J[pair*q*q + k];
                    }
                    barrier(CLK_LOCAL_MEM_FENCE);

                    seqm = getbyte(&sbm, cm);
                    energy = energy + lcouplings[q*seqn + seqm];
                    barrier(CLK_LOCAL_MEM_FENCE);
                    
                    pair++;
                }
            }
        }
    }
    return energy;
}

// this function expects J in "unpacked" form, with L*L*q*q elements
inline float getEnergiesfX(__global float *J,
                          __global uint *seqmem,
                                   uint  buflen,
                          __local float *lcouplings){
    // This function is complicated by optimizations for the GPU.
    // For clarity, here is equivalent but clearer (pseudo)code:
    //
    //float energy = 0;
    //for(n = 0; n < L-1; n++){
    //    for(m = n+1; m < L; m++){
    //       energy += J[n,m,seq[n],seq[m]];
    //    }
    //}

    uint li = get_local_id(0);

    uint seqm, seqn, seqp;
    uint cn, cm;
    uint sbn, sbm;
    uint n,m,k;
    float energy = 0;
    n = 0;
    while(n < L-1){
        uint sbn = seqmem[(n/4)*buflen + get_global_id(0)];
        #pragma unroll //probably ignored
        for(cn = n%4; cn < 4 && n < L-1; cn++, n++){
            seqn = getbyte(&sbn, cn);
            m = n+1;
            while(m < L){
                uint sbm = seqmem[(m/4)*buflen + get_global_id(0)];
                
                #pragma unroll
                for(cm = m%4; cm < 4 && m < L; cm++, m++){
                    for(k = li; k < q*q; k += get_local_size(0)){
                        lcouplings[k] = J[(n*L + m)*q*q + k];
                    }
                    barrier(CLK_LOCAL_MEM_FENCE);

                    seqm = getbyte(&sbm, cm);
                    energy = energy + lcouplings[q*seqn + seqm];
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
            }
        }
    }
    return energy;
}

__kernel
void getEnergies(__global float *J,
                 __global uint *seqmem,
                          uint  buflen,
                 __global float *energies){
    __local float lcouplings[4*q*q];
    energies[get_global_id(0)] = getEnergiesf(J, seqmem, buflen, lcouplings);
}

#define SWAPF(a,b) {float tmp = a; a = b; b = tmp;}

// ****************************** Metropolis sampler **************************

__kernel
void initRNG2(__global mwc64xvec2_state_t *rngstates,
                       ulong offset,
                       ulong nsamples){
    mwc64xvec2_state_t rstate;
    MWC64XVEC2_SeedStreams(&rstate, offset, nsamples);
    rngstates[get_global_id(0)] = rstate;
}

inline float UpdateEnergy(__local float *lcouplings, __global float *J,
                          global uint *seqmem, uint nseqs,
                          uint pos, uint seqp, uchar mutres, float energy){
    uint m = 0;
    while(m < L){
        //loop through seq, changing energy by changed coupling with pos

        //load the next 4 rows of couplings to local mem
        uint n;
        for(n = get_local_id(0); n < min((uint)4, L-m)*q*q;
                                                      n += get_local_size(0)){
            lcouplings[n] = J[(pos*L + m)*q*q + n];
        }

        //this line is the bottleneck of the entire MCMC analysis
        // (would some kind of prefetch help here?)
        uint sbm = seqmem[(m/4)*nseqs + get_global_id(0)];

        barrier(CLK_LOCAL_MEM_FENCE); // for lcouplings and sbm

        //calculate contribution of those 4 rows to energy
        for(n = 0; n < 4 && m < L; n++, m++){
            if(m == pos){
                continue;
            }
            uint seqm = getbyte(&sbm, n);
            energy += lcouplings[q*q*n + q*mutres + seqm];
            energy -= lcouplings[q*q*n + q*seqp   + seqm];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    return energy;
}

__kernel
void metropolis(__global float *J,
                __global mwc64xvec2_state_t *rngstates,
                __global uint *position_list,
                         uint nsteps, // must be multiple of L
                __global float *energies, //ony used to measure fp error
                __global float *betas,
                __global uint *seqmem){

    uint nseqs = get_global_size(0);
    mwc64xvec2_state_t rstate = rngstates[get_global_id(0)];

    //set up local mem
    __local float lcouplings[q*q*4];
    __local uint shared_position;

    //initialize energy
    float energy = getEnergiesfX(J, seqmem, nseqs, lcouplings);
    float B = betas[get_global_id(0)];

    uint i;
    for(i = 0; i < nsteps; i++){
        uint pos = position_list[i];
        uint2 rng = MWC64XVEC2_NextUint2(&rstate);
        uint mutres = rng.x%q;  // small error here if MAX_INT%q != 0
                                 // of order q/MAX_INT in marginals
        uint sbn = seqmem[(pos/4)*nseqs + get_global_id(0)];
        uint seqp = getbyte(&sbn, pos%4);

        float newenergy = UpdateEnergy(lcouplings, J, seqmem, nseqs,
                                       pos, seqp, mutres, energy);

        //apply MC criterion and possibly update
        if(exp(-B*(newenergy - energy)) > uniformMap(rng.y)){
            setbyte(&sbn, pos%4, mutres);
            seqmem[(pos/4)*nseqs + get_global_id(0)] = sbn;
            energy = newenergy;
        }
    }

    rngstates[get_global_id(0)] = rstate;

#ifdef MEASURE_FP_ERROR
    energies[get_global_id(0)] = energy;
#endif
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
    while(j <= gi){
        i++;
        j += L-1-i;
    }
    j = gi + L - j; //careful with underflow!

    for(n = 0; n < q*q; n++){
        hist[nhist*n + li] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    uint tmp;
    // loop through all sequences
    for(n = li; n < nseq; n += nhist){
        tmp = seqmem[(i/4)*buflen + n];
        uint seqi = ((uchar*)&tmp)[i%4];
        tmp = seqmem[(j/4)*buflen + n];
        uint seqj = ((uchar*)&tmp)[j%4];
        hist[nhist*(q*seqi + seqj) + li]++;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    //merge histograms
    for(n = 0; n < q*q; n++){
        m = nhist;
        while (m > 1) {
            uint odd = m%2;
            m = (m+1)>>1; //div by 2 rounded up
            if(li < m - odd){
                hist[nhist*n + li] = hist[nhist*n + li] + hist[nhist*n + li +m];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }

    for(n = li; n < q*q; n += nhist){ //only loops once if nhist > q*q
        bicount[gi*q*q + n] = hist[nhist*n];
    }
}

__kernel
void bicounts_to_bimarg(__global uint *bicount,
                        __global float *bimarg,
                                 uint  nseq) {
    uint n = get_global_id(0);
    if(n > NCOUPLE){
        return;
    }

    bimarg[n] = ((float)bicount[n])/((float)nseq);
}

__kernel //call with group size = NHIST, for nPair groups
void countMarkedBivariate(__global uint *bicount,
                          uint nseq,
                 __global uint *marks,
                 __global uint *seqmem,
                 __local  uint *hist) {
    uint li = get_local_id(0);
    uint gi = get_group_id(0);
    uint nhist = get_local_size(0);
    uint i,j,n,m;

    // only meant to be called for small buffer so nseq == buflen

    //figure out which i,j pair we are
    i = 0;
    j = L-1;
    while(j <= gi){
        i++;
        j += L-1-i;
    }
    j = gi + L - j; //careful with underflow!

    for(n = 0; n < q*q; n++){
        hist[nhist*n + li] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    uint tmp;
    //loop through all sequences
    for(n = li; n < nseq; n += nhist){
        if(marks[n] != -1){
            tmp = seqmem[(i/4)*nseq+n];
            uint seqi = ((uchar*)&tmp)[i%4];
            tmp = seqmem[(j/4)*nseq+n];
            uint seqj = ((uchar*)&tmp)[j%4];
            hist[nhist*(q*seqi + seqj) + li]++;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    //merge histograms
    for(n = 0; n < q*q; n++){
        m = nhist;
        while (m > 1) {
            uint odd = m%2;
            m = (m+1)>>1; //div by 2 rounded up
            if(li < m - odd){
                hist[nhist*n + li] = hist[nhist*n + li] + hist[nhist*n + li +m];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }

    for(n = li; n < q*q; n += nhist){ //only loops once if nhist > q*q
        bicount[gi*q*q + n] = hist[nhist*n];
    }
}

__kernel //call with global work size = to # sequences
void perturbedWeights(__global float *J,
                      __global uint *seqmem,
                               uint  buflen,
                      __global float *weights,
                      __global float *energies){
    __local float lcouplings[q*q];
    float energy = getEnergiesf(J, seqmem, buflen, lcouplings);
    weights[get_global_id(0)] = exp(-(energy - energies[get_global_id(0)]));
}

__kernel //simply sums a vector. Call with single group of size VSIZE, which must be power of two
void sumFloats(__global float *data,
               __global float *output,
                        uint  len,
               __local  float *sums){
    uint li = get_local_id(0);
    uint vsize = get_local_size(0);
    uint n;

    // accumulate through array
    sums[li] = 0;
    for(n = li; n < len; n += vsize){
        sums[li] += data[n];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    //reduce accumulated sums (req power of two unlike elsewhere in this file)
    for(n = vsize/2; n > 0; n >>= 1){
        if(li < n){
            sums[li] = sums[li] + sums[li + n];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    // assumes li is even. Need to add last element if odd
    if(li == 0){
        *output = sums[0];
    }
}

// Idea: Have that NHIST, and HISTWS (work-size) are powers of 2, with
// HISTWS >= NHIST. Then in each loop over n below, we load memory as:
//      <-----HISTWS----->
// si   xxxxxxxxxxxxxxxxxx  (remember these are packed uints, so expand 4x)
// sj   xxxxxxxxxxxxxxxxxx
//  w   xxxxxxxxxxxxxxxxxx xxxxxxxxxxxxxxxxxx xxxxxxxxxxxxxxxxxx xxxxxxxxxxxxxxxxxx 
//      <-NHIST->
//
// In each loop load si, sj, and the 1st w segment. Then loop over all windows
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
    while(j <= gi){
        i++;
        j += L-1-i;
    }
    j = gi + L - j; //careful with underflow!
    
    for(n = li; n < NHIST*q*q; n += HISTWS){
        hist[n] = 0;
    }

    //loop through all sequences
    for(n = li; n < nseq/4; n += HISTWS){
        sid[li] = seqmem[i*buflen + n];
        sjd[li] = seqmem[j*buflen + n];
        #pragma unroll
        for(uint k = 0; k < 4; k++){
            w[li] = weights[HISTWS*(4*(n/HISTWS) + k) + li];
            barrier(CLK_LOCAL_MEM_FENCE);
            if(li < NHIST) {
                for(uint l = li; l < HISTWS; l += NHIST) {
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
    for(n = li/(NHIST/2); n < q*q; n += HISTWS/(NHIST/2)){
        // since NHIST is pow of two we can use simpler reduction code (no odd)
        for(m = NHIST/2; m > 0; m >>= 1){
            if(x < m){
                hist[NHIST*n + x] = hist[NHIST*n + x] + hist[NHIST*n + x + m];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }

    for(n = li; n < q*q; n += HISTWS) {
        bimarg_new[gi*q*q + n] = hist[NHIST*n];
    }
}

__kernel
void addBiBufs(__global float *dst, __global float *src){
    uint n = get_global_id(0);
    if(n > NCOUPLE){
        return;
    }
    dst[n] += src[n];
}

// expects to be called with work-group size of q*q
// Local scratch memory must be provided:
// sums is q*q elements
float sumqq(float v, uint li, __local float *scratch){
    uint m;

    scratch[li] = v;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // reduction loop which accounts for odd vector sizes
    m = q*q;
    while (m > 1) {
        uint odd = m%2;
        m = (m+1)>>1; //div by 2 rounded up
        if(li < m - odd){
            scratch[li] = scratch[li] + scratch[li + m];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    return scratch[0];
}

// expects to be called with work-group size of q*q
__kernel
void renormalize_bimarg(__global float *bimarg){
    __local float scratch[q*q];
    float bim = bimarg[get_global_id(0)];
    bimarg[get_global_id(0)] = bim/sumqq(bim, get_local_id(0), scratch);
}

// expects to be called with work-group size of q*q
// Local scratch memory must be provided:
// scratch is q*q elements, hi and hj are q elements.
float zeroGauge(float J, uint li, __local float *scratch,
                __local float *hi, __local float *hj){
    uint m;

    // next couple lines are essentially a transform to the "zero" gauge

    //add up rows
    scratch[li] = J;
    barrier(CLK_LOCAL_MEM_FENCE);
    m = q;
    while (m > 1) {
        uint odd = m%2;
        m = (m+1)>>1; //div by 2 rounded up
        if(li%q < m - odd){
            scratch[li] = scratch[li] + scratch[li + m];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if(li < q){
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
        if(li%q < m - odd){
            scratch[li] = scratch[li] + scratch[li + m];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if(li < q){
        hj[li] = scratch[q*li]/q;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    //get total sum
    m = q;
    while (m > 1) {
        uint odd = m%2;
        m = (m+1)>>1; //div by 2 rounded up
        if(li < m - odd){
            scratch[q*li] = scratch[q*li] + scratch[q*(li + m)];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    float mean = scratch[0]/q/q;
    barrier(CLK_LOCAL_MEM_FENCE);

    if(li < q){
        hi[li] = hi[li] - mean;
        hj[li] = hj[li] - mean;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    return J - (hi[li/q] + hj[li%q]) - mean;
}

float fbnorm(float J, uint li, __local float *sums){
    return sqrt(sumqq(J*J, li, sums));
}


__kernel
void updatedJ(__global float *bimarg_target,
              __global float *bimarg,
                       float gamma,
                       float pc,
              __global float *Ji,
              __global float *Jo){
    uint n = get_global_id(0);

    if(n > NCOUPLE){
        return;
    }

    Jo[n] = Ji[n] - gamma*(bimarg_target[n] - bimarg[n])/(bimarg[n] + pc);
}

__kernel
void updatedJ_l2z(__global float *bimarg_target,
                  __global float *bimarg,
                           float gamma,
                           float pc,
                           float lh, float lJ,
                  __global float *Ji,
                  __global float *Jo){
    uint li = get_local_id(0);
    uint gi = get_group_id(0);
    uint n = gi*q*q + li;

    __local float hi[q], hj[q];
    __local float scratch[q*q];

    float J0 = zeroGauge(Ji[n], li, scratch, hi, hj);
    float R = lJ*J0 + lh*(hi[li/q] + hj[li%q]);
    Jo[n] = Ji[n] - gamma*(bimarg_target[n] - bimarg[n] + R)/(bimarg[n] + pc);
}

__kernel
void updatedJ_X(__global float *bimarg_target,
                __global float *bimarg,
                __global float *Creg,
                         float gamma,
                         float pc,
                __global float *Ji,
                __global float *Jo){
    uint li = get_local_id(0);
    uint gi = get_group_id(0);
    uint n = gi*q*q + li;

    __local float l_qq[q*q];
    float X = sumqq(Ji[n]*Creg[n], li, l_qq);
    float bias = Creg[n]*sign(X);
    //float bias = -Creg[n];  // equivalent to pre_regularize

    Jo[n] = Ji[n] - gamma*(bimarg_target[n] - bimarg[n] + bias)/(bimarg[n] + pc);
}

// expects to be called with work-group size of q*q
float getXC(float J, float bimarg, uint li, __local float *scratch,
            __local float *fi, __local float *fj){
    uint m;

    // add up bimarg rows
    scratch[li] = bimarg;
    barrier(CLK_LOCAL_MEM_FENCE);
    m = q;
    while (m > 1) {
        uint odd = m%2;
        m = (m+1)>>1; //div by 2 rounded up
        if(li%q < m - odd){
            scratch[li] = scratch[li] + scratch[li + m];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if(li < q){
        fi[li] = scratch[q*li]/q;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    //add up bimarg columns
    scratch[q*(li%q) + li/q] = bimarg;
    barrier(CLK_LOCAL_MEM_FENCE);
    m = q;
    while (m > 1) {
        uint odd = m%2;
        m = (m+1)>>1; //div by 2 rounded up
        if(li%q < m - odd){
            scratch[li] = scratch[li] + scratch[li + m];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if(li < q){
        fj[li] = scratch[q*li]/q;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // each work unit computes one of the Xijab terms
    scratch[li] = J*(bimarg - fi[li/q]*fj[li%q]);
    barrier(CLK_LOCAL_MEM_FENCE);

    //sum up all terms over ab
    m = q;
    while (m > 1) {
        uint odd = m%2;
        m = (m+1)>>1; //div by 2 rounded up
        if(li < m - odd){
            scratch[q*li] = scratch[q*li] + scratch[q*(li + m)];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // write out C and return X
    scratch[li] = bimarg - fi[li/q]*fj[li%q];
    barrier(CLK_LOCAL_MEM_FENCE);
    return scratch[0];
}

__kernel
void updatedJ_Xself(__global float *bimarg_target,
              __global float *bimarg,
              __global float *lambdas,
                       float gamma,
                       float pc,
              __global float *Ji,
              __global float *Jo){
    uint li = get_local_id(0);
    uint gi = get_group_id(0);
    uint n = gi*q*q + li;

    __local float hi[q], hj[q];
    __local float C[q*q];

    float X = getXC(Ji[n], bimarg[n], li, C, hi, hj);
    float bias = lambdas[gi]*C[li]*sign(X);

    Jo[n] = Ji[n] - gamma*(bimarg_target[n] - bimarg[n] + bias)/(bimarg[n]+pc);
}



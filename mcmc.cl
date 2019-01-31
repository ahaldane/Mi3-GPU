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
void packfV(__global float *v,
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

//only call from kernels with nseqs work units!!!!!!
inline float getEnergiesf(__global float *J,
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

    uchar seqm, seqn, seqp;
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
                    // could add skip for fixpos here...
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
                          uint pos, uchar seqp, uchar mutres, float energy){
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
        // (could consider using prefetch here)
        uint sbm = seqmem[(m/4)*nseqs + get_global_id(0)];

        barrier(CLK_LOCAL_MEM_FENCE); // for lcouplings and sbm

        //calculate contribution of those 4 rows to energy
        for(n = 0; n < 4 && m < L; n++, m++){
            if(m == pos){
                continue;
            }
            uchar seqm = getbyte(&sbm, n);
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
    float energy = getEnergiesf(J, seqmem, nseqs, lcouplings);
    float B = betas[get_global_id(0)];

    uint i;
    for(i = 0; i < nsteps; i++){
        uint pos = position_list[i];
        uint2 rng = MWC64XVEC2_NextUint2(&rstate);
        uchar mutres = rng.x%q;  // small error here if MAX_INT%q != 0
                                 // of order q/MAX_INT in marginals
        uint sbn = seqmem[(pos/4)*nseqs + get_global_id(0)];
        uchar seqp = getbyte(&sbn, pos%4);

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

    //once we get to use atomic operations in openCL 2.0,
    //this might be sped up by having a single shared histogram,
    //with a much larger work group size.

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
        uchar seqi = ((uchar*)&tmp)[i%4];
        tmp = seqmem[(j/4)*buflen + n];
        uchar seqj = ((uchar*)&tmp)[j%4];
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

    //once we get to use atomic operations in openCL 2.0,
    //this might be sped up by having a single shared histogram,
    //with a much larger work group size.

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
            uchar seqi = ((uchar*)&tmp)[i%4];
            tmp = seqmem[(j/4)*nseq+n];
            uchar seqj = ((uchar*)&tmp)[j%4];
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
    __local float lcouplings[4*q*q];
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

__kernel //call with group size = NHIST, for nPair groups
void weightedMarg(__global float *bimarg_new,
                  __global float *weights,
                  __global float *sumweights,
                           uint nseq,
                  __global uint *seqmem,
                           uint  buflen,
                  __local  float *hist) {
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
    //loop through all sequences
    for(n = li; n < nseq; n += nhist){
        tmp = seqmem[(i/4)*buflen + n];
        uchar seqi = getbyte(&tmp, i%4);
        tmp = seqmem[(j/4)*buflen + n];
        uchar seqj = getbyte(&tmp, j%4);
        hist[nhist*(q*seqi + seqj) + li] += weights[n];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    //merge histograms
    for(n = 0; n < q*q; n++){
        m = nhist;
        while (m > 1) {
            uint odd = m%2;
            m = (m+1)>>1; //div by 2 rounded up
            if(li < m - odd){
                hist[nhist*n + li] = hist[nhist*n + li] + hist[nhist*n + li + m];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }

    for(n = li; n < q*q; n += nhist){ //only loops once if nhist >= q*q
        bimarg_new[gi*q*q + n] = hist[nhist*n];///(*sumweights);
    }
}

// bivariate count kernel, optimized to take advantage of 4-byte transposed
// sequence memory. 
//
// Seems to be slower than the kernel above even though it seemed more 
// memory-efficient. Keeping it for now for testing purposes.
//
// group size must be a multiple of 256. Each wg loads 16 uint pieces of
// sequence memory: 16 uints of row i4 of seqmem, 16 uints of row j4 of
// sequmem, and 16 floats of weights. Each uint contains 4 characters. The 256
// work-units are divided into 16 groups of 16 units. Each group processes one
// of the 16 words of sequence memory. Within each group of 16, each wu takes
// care of one of the 16 i,j combinations for that word, computing which
// histogram entry to increment.
//
// There are 16 histograms. Everything is arranged so that in each round, all
// the histograms can be updated simultaneously by one of the 16 groups. We
// need to do 16 loops of this, one per group, since the histogram updates need
// to be atomic across groups.
__kernel 
void weightedMargAlt(__global float *bimarg_new,
                      __global float *weights,
                                uint  nseq,
                       __global uint *seqmem,
                                uint  buflen) {
    uint li = get_local_id(0);
    uint gi = get_group_id(0);
    uint i4, j4;

    //figure out which reduced i4/j4 pair we are
    i4 = 0;
    j4 = SWORDS;
    while(j4 <= gi){
        i4++;
        j4 += SWORDS-i4;
    }
    j4 = gi + SWORDS - j4; //careful with underflow!

    __local uint seq_rowi4[16];
    __local uint seq_rowj4[16];
    __local float tmpweight[16];
    __local float hist[16*q*q];

    //initialize 16 histograms, for every combination of 4x4 positions. Note:
    // the histograms are interleaved in memory, see below.
    for(uint n = li; n < 16*q*q; n += 256){
        hist[n] = 0;
    }

    //loop through all sequences in chunks of 16
    for(uint n = li; n < nseq; n += 16){
        if (li < 16) {
            seq_rowi4[li] = seqmem[i4*buflen + n];
        }
        else if (li < 32) {
            seq_rowj4[li - 16] = seqmem[j4*buflen + n - 16];
        }
        else if (li < 48) {
            tmpweight[li - 32] = weights[n - 32];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        
        float w = tmpweight[li/16];
        uint si4 = seq_rowi4[li/16];
        uint sj4 = seq_rowj4[li/16];
        uint si = getbyte(&si4, (li%16)/4);
        uint sj = getbyte(&sj4, li%4);
        uint ind = 16*(q*si + sj) + (li%16);

        for(uint x = 0; x < 16; x++) {
            if (li/16 == x) {
                hist[ind] += w;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        // Note arrangement of histogram elements Hijab, where ij
        // are the position indices (0 to 3) in si and sj, and ab
        // are the residue letters
        //                    <--  16  -->
        //  ^   H00AA H01AA H02AA H03AA H10AA H11AA H12AA ...
        //  |   H00AB H01AB H02AB H03AB ...
        // q*q  H00AC H01AC ...
        //  |   H00BA
        //  v   ...
        //                     (shown in C-order)
        // The work units update successive columns in this table in each
        // round, rotating through columns over rounds ij
    }

    // write back histograms. requires transpose and map to ij index
    for(uint n = li; n < 16*q*q; n += 256){
        // a computation to convert a i,j pair to bimarg row
        uint i = ((n/(q*q))/4) + 4*i4;
        uint j = ((n/(q*q))%4) + 4*j4;
        // Note ordering of work units to ij pairs:
        //  n   0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15
        // ij  00 01 02 03 10 11 12 13 20 21 22 23 30 31 32 33
        if(i >= j || j >= L) { //only happens if rowi == rowj or in end padding
            continue;
        }
        uint out_row = (L*(L-1)/2) - (L-i)*((L-i)-1)/2 + j - i - 1;
        float f = hist[16*(n%(q*q)) + (n/(q*q))];
        bimarg_new[q*q*out_row + (n%(q*q))] = f;
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
// zeroj is q*q elements, rsums and csums are q elements.
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
    //float bias = -Creg[n];

    Jo[n] = Ji[n] - gamma*(bimarg_target[n] - bimarg[n] + bias)/(bimarg[n] + pc);
}




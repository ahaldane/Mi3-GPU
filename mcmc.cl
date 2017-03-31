//Copyright 2016 Allan Haldane.
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

//typedef unsigned char uchar;
//typedef unsigned int uint;

float uniformMap(uint i){
    return (i>>8)*0x1.0p-24f; //converts a 32 bit integer to a float [0,1)
}

#define NCOUPLE ((L*(L-1)*q*q)/2)

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

    #define SWORDS ((L-1)/4+1)
    for(w = 0; w < SWORDS; w++){
        largebuf[w*nlargebuf + offset + n] = smallbuf[w*nseqs + n];
    }
    #undef SWORDS
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

    #define SWORDS ((L-1)/4+1)
    for(w = 0; w < SWORDS; w++){
        if(inds[n] != -1){
            largebuf[w*nlarge + offset + inds[n]] = smallbuf[w*nseqs + n];
        }
    }
    #undef SWORDS
}

__kernel
void restoreSeqs(__global uint *smallbuf,
                 __global uint *largebuf,
                          uint  nlargebuf,
                          uint  offset){
    uint w, n;
    uint nseqs = get_global_size(0);

    #define SWORDS ((L-1)/4+1)
    for(w = 0; w < SWORDS; w++){
        for(n = get_local_id(0); n < nseqs; n += get_local_size(0)){
            smallbuf[w*nseqs + n] = largebuf[w*nlargebuf + offset + n];
        }
    }
    #undef SWORDS
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
    //double energy = 0;
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

__kernel //__attribute__((work_group_size_hint(WGSIZE, 1, 1)))
void getEnergies(__global float *J,
                 __global uint *seqmem,
                          uint  buflen,
                 __global float *energies){
    __local float lcouplings[4*q*q];
    energies[get_global_id(0)] = getEnergiesf(J, seqmem, buflen, lcouplings);
}

#define SWAPF(a,b) {float tmp = a; a = b; b = tmp;}

/*
inline float logZE(__global float *J, int offset){
    uint seqm, seqn;
    uint n,m,k;
    float energy = 0;
    seqn = get_global_id(0) + offset;
    k = 0;
    for(n = 0; n < L-1; n++){
        seqm = seqn/q;
        for(m = n+1; m < L; m++){
            // this loads q*q floats at a time. Might be sped up by loading
            // multiple rows at once to shared like in getEnergies?
            // Does this cause each warp to load the same block?
            // XXX also, should the J matrix rows be aligned by padding?
            // or can we overcome this with dummy padding in the local mem
            // on the other hand we have no memory barriers and caching may help
            energy += J[k*q*q + q*(seqn%q) + (seqm%q)];
            seqm = seqm/q;
            k++;
        }
        seqn = seqn/q;
    }
    return energy;
}

inline float logZE_prefetch(__global float *J, int offset){
    __local float localJ[WGSIZE + q*q];
    int rowsLoaded = WGSIZE/(q*q);
    int leftover = WGSIZE - rowsLoaded*q*q;
    int fetchind;
    float prefetch = J[li];

    uint seqm, seqn;
    uint n,m;

    float energy = 0;
    seqn = get_global_id(0) + offset;
    seqm = seqn/q;
    n = 0;
    for(fetchind = li + WGSIZE; ; fetchind += WGSIZE){
        // put prefetchd data in local mem
        barrier(CLK_LOCAL_MEM_FENCE);
        if(WGSIZE%(q*q) != 0){
            if(li < leftover){ //move leftover to front
                localJ[li] = localJ[rowsLoaded*q*q + li];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        localJ[leftover + li] = prefetch; 
        barrier(CLK_LOCAL_MEM_FENCE);

        // prefetch next Js (except last loop)
        if(fetchind < NCOUPLE){
            prefetch = J[fetchind];
        }
        // rest of this block should execute while prefetch is loading

        rowsLoaded = (leftover + WGSIZE)/(q*q);
        leftover = (leftover + WGSIZE) - rowsLoaded*q*q;
        
        uint k;
        for(k = 0; k < rowsLoaded; k++){
            energy += localJ[k*q*q + q*(seqn%q) + (seqm%q)];

            m++;
            seqm = seqm/q;
            if(m == L){
                n++;
                seqn = seqn/q;
                if(n == L-1){
                    return energy;
                }
                m = n+1;
                seqm = seqn/q;
            }
        }
    }
}


inline float logZE_prefetch2(__global float *J, int offset){
    uint seqm, seqn;
    uint n,m,k=0;
    __local float localJ[WGSIZE + q*q];
    int lenLoaded = 0;
    int leftover = 0;
    int fetch = 0;
    float prefetch = J[li];

    float energy = 0;
    seqn = get_global_id(0) + offset;
    k = 0;
    for(n = 0; n < L-1; n++){
        seqm = seqn/q;
        for(m = n+1; m < L; m++){
            if(k == rowsLoaded){
                // put prefetchd data in local mem
                barrier(CLK_LOCAL_MEM_FENCE);
                int leftover = lenLoaded - rowsLoaded*q*q;
                if(li < leftover){
                    localJ[li] = localJ[li + rowsLoaded*q*q];
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                localJ[li+leftover] = prefetch; 
                barrier(CLK_LOCAL_MEM_FENCE);
                lenLoaded = leftover + WGSIZE;
                rowsLoaded = lenLoaded/(q*q);

                // prefetch
                prefetch = J[li + fetch*WGSIZE]; // needs padding to WGSIZE
                fetch++;
                k = 0;
            }

            energy += localJ[k*q*q + q*(seqn%q) + (seqm%q)];
            seqm = seqm/q;
            k++;
            rowsleft--;
        }
        seqn = seqn/q;
    }
    return energy;
}

inline float enumerate_logZ(__global float *J){
    float F = logZE(0);  // free energy
    float E = F;         // average energy
    int i;
    for(i = 1; i < nloops; i++){
        float energy = logZE_prefetch(i*get_global_size(0), scratch);
        float dF = F - energy;
        if(dF > 0){
            dF = -dF;
            F = energy;
            SWAPF(E, energy);
        }
        float expdF = exp(dF);
        F = F - log1p(expdF);
        E = (E + energy*expdF)/(1+expdF);
    }

    // sum up E/F in workgroup
    for(n = vsize/2; n > 0; n >>= 1){
        if(li > n && li < 2*n){
            // use half of scratch for E, half for F
            scratch[li-n] = F;
            scratch[li] = E;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if(li < n){
            float F2 = scratch[li];
            float E2 = scratch[li+n];
            float dF = F - F2;
            if(dF > 0){
                dF = -dF;
                F = F2;
                SWAPF(E,E2);
            }
            float expdF = exp(dF);
            F = F - log1p(expdF);
            E = (E + E2*expdF)/(1+expdF);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if(li == 0){
        outF[gi] = F;
        outE[gi] = E;
    }
}
*/

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
        // XXX could consider using prefetch here
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

__kernel //__attribute__((work_group_size_hint(WGSIZE, 1, 1)))
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

// ****************************** Gibbs sampler **************************

__kernel
void initRNG(__global mwc64x_state_t *rngstates,
                      ulong offset,
                      ulong nsamples){
    mwc64x_state_t rstate;
    MWC64X_SeedStreams(&rstate, offset, nsamples);
    rngstates[get_global_id(0)] = rstate;
}

inline void GibbsProb(__local float *lcouplings, __global float *J,
                      __global uint *seqmem, uint nseqs, uint pos,
                      float *prob){
    uint o;
    for(o = 0; o < q; o++){
        prob[o] = 0;
    }

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
        uint sbm = seqmem[(m/4)*nseqs + get_global_id(0)];

        barrier(CLK_LOCAL_MEM_FENCE); // for lcouplings and sbm

        //calculate contribution of those 4 rows to energy
        for(n = 0; n < 4 && m < L; n++, m++){
            if(m == pos){
                continue;
            }
            uchar seqm = getbyte(&sbm, n);
            for(o = 0; o < q; o++){
                prob[o] += lcouplings[q*q*n + q*o + seqm]; //bank conflict?
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    float Z = 0;
    for(o = 0; o < q; o++){
        Z += exp(-prob[o]);
        prob[o] = Z;
    }
    for(o = 0; o < q; o++){
        prob[o] = prob[o]/Z;
    }
}

__kernel //__attribute__((work_group_size_hint(WGSIZE, 1, 1)))
void gibbs(__global float *J,
           __global mwc64x_state_t *rngstates,
           __global uint *position_list,
                    uint nsteps, // must be multiple of L
           __global float *energies, //ony used to measure fp error
           __global uint *seqmem){

    uint nseqs = get_global_size(0);
    mwc64x_state_t rstate = rngstates[get_global_id(0)];

    //set up local mem
    __local float lcouplings[q*q*4];
    float gibbsprob[q];

    uint i;
    for(i = 0; i < nsteps; i++){
        uint pos = position_list[i];

        GibbsProb(lcouplings, J, seqmem, nseqs, pos, gibbsprob);
        float p = uniformMap(MWC64X_NextUint(&rstate));
        uint o = 0;
        while(o < q-1 && gibbsprob[o] < p){
            o++;
        }

        // if we had a byte addressable store, could simply assign
        uint sbn = seqmem[(pos/4)*nseqs + get_global_id(0)];
        setbyte(&sbn, pos%4, o);
        seqmem[(pos/4)*nseqs + get_global_id(0)] = sbn;
    }

    rngstates[get_global_id(0)] = rstate;
}

// ****************************** Histogram Code **************************

__kernel //call with group size = NHIST, for nPair groups
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
        for(m = nhist/2; m > 0; m >>= 1){
            if(li < m){
                hist[nhist*n + li] = hist[nhist*n + li] + hist[nhist*n + li +m];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }

    for(n = li; n < q*q; n += nhist){ //only loops once if nhist > q*q
        bicount[gi*q*q + n] = hist[nhist*n];
    }
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
        for(m = nhist/2; m > 0; m >>= 1){
            if(li < m){
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

__kernel //simply sums a vector. Call with single group of size VSIZE
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
    //reduce accumulated sums
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
        uchar seqi = ((uchar*)&tmp)[i%4];
        tmp = seqmem[(j/4)*buflen + n];
        uchar seqj = ((uchar*)&tmp)[j%4];
        hist[nhist*(q*seqi + seqj) + li] += weights[n];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    //merge histograms
    for(n = 0; n < q*q; n++){
        for(m = nhist/2; m > 0; m >>= 1){
            if(li < m){
                hist[nhist*n + li] = hist[nhist*n + li] + hist[nhist*n + li +m];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }

    for(n = li; n < q*q; n += nhist){ //only loops once if nhist >= q*q
        bimarg_new[gi*q*q + n] = hist[nhist*n]/(*sumweights);
    }
}

__kernel
void updatedJ(__global float *bimarg_target,
              __global float *bimarg,
                       float gamma,
                       float pc,
              __global float *J_orig,
              __global float *Ji,
              __global float *Jo){
    uint n = get_global_id(0);

    if(n > NCOUPLE){
        return;
    }

    Jo[n] = Ji[n] - gamma*(bimarg_target[n] - bimarg[n])/(bimarg[n] + pc);
}

// expects to be called with work-group size of q*q
// Local scratch memory must be provided:
// zeroj is q*q elements, rsums and csums are q elements.
float zeroGauge(float J, uint li, __local float *zeroj,
                __local float *rsums, __local float *csums){
    uint m;

    // next couple lines are essentially a transform to the "zero" gauge

    //add up rows
    zeroj[li] = J;
    barrier(CLK_LOCAL_MEM_FENCE);
    for(m = q/2; m > 0; m >>= 1){
        if(li%q < m){
            zeroj[li] = zeroj[li] + zeroj[li + m];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if(li < q){
        rsums[li] = zeroj[q*li];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    //add up columns
    zeroj[q*(li%q) + li/q] = J;
    barrier(CLK_LOCAL_MEM_FENCE);
    for(m = q/2; m > 0; m >>= 1){
        if(li%q < m){
            zeroj[li] = zeroj[li] + zeroj[li + m];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if(li < q){
        csums[li] = zeroj[q*li];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    //get total sum
    for(m = q/2; m > 0; m >>= 1){
        if(li < m){
            zeroj[q*li] = zeroj[q*li] + zeroj[q*(li + m)];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float total = zeroj[0];
    barrier(CLK_LOCAL_MEM_FENCE);
    //compute zero gauge J
    return J - (rsums[li/q] + csums[li%q])/q + total/(q*q);
}

__kernel
void updatedJ_L2(__global float *bimarg_target,
              __global float *bimarg,
                       float gamma,
                       float pc,
              __global float *J_orig,
              __global float *Ji,
              __global float *Jo){
    uint n = get_global_id(0);
    int i = li%q;
    int j = li/q;

    __local float hi[q], hj[q];
    __local float scratch[q*q];

    float J = zeroGauge(Ji[n], li, scratch, fi, fj);

    if(n > NCOUPLE){
        return;
    }

    Jo[n] = Ji[n] - gamma*(bimarg_target[n] - bimarg[n])/(bimarg[n] + pc);
}

// expects to be called with work-group size of q*q
// Local scratch memory must be provided:
// sums is q*q elements
float fbnorm(float J, uint li, __local float *sums){
    uint m;

    sums[li] = J*J;
    barrier(CLK_LOCAL_MEM_FENCE);
    for(m = q*q/2; m > 0; m >>= 1){
        if(li < m){
            sums[li] = sums[li] + sums[li + m];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    return sqrt(sums[0]);
}

float sumqq(float v, uint li, __local float *scratch){
    uint m;

    scratch[li] = v;
    barrier(CLK_LOCAL_MEM_FENCE);
    for(m = q*q/2; m > 0; m >>= 1){
        if(li < m){
            scratch[li] = scratch[li] + scratch[li + m];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    // XXX assumes q is even. Need to add last element if odd

    return scratch[0];
}

__kernel
void updatedJ_weightfn(__global float *bimarg_target,
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

    Jo[n] = Ji[n] - gamma*(bimarg_target[n] - bimarg[n] + bias)/(bimarg[n] + pc);
}

// expects to be called with work-group size of q*q
// Local scratch memory must be provided:
// zeroj is q*q elements, rsums and csums are q elements.
float get_unimarg(float ff, uint li,
                  __local float *fi,
                  __local float *fj, 
                  __local float *scratch){
    uint m;

    //add up rows
    barrier(CLK_LOCAL_MEM_FENCE);
    scratch[li] = ff;
    barrier(CLK_LOCAL_MEM_FENCE);
    for(m = q/2; m > 0; m >>= 1){
        if(li%q < m){
            scratch[li] = scratch[li] + scratch[li + m];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if(li < q){
        fi[li] = scratch[q*li];
    }
    //add up columns
    barrier(CLK_LOCAL_MEM_FENCE);
    scratch[q*(li%q) + li/q] = ff;
    barrier(CLK_LOCAL_MEM_FENCE);
    for(m = q/2; m > 0; m >>= 1){
        if(li%q < m){
            scratch[li] = scratch[li] + scratch[li + m];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if(li < q){
        fj[li] = scratch[q*li];
    }
}

//this kernel is quite inefficient, but it's not a bottleneck...
__kernel
void updatedJ_Lstep(__global float *bimarg_target,
              __global float *bimarg,
                       float gamma,
                       float pc,
              __global float *Ji,
              __global float *Jo){
    uint li = get_local_id(0);
    uint gi = get_group_id(0);
    uint n = gi*q*q + li;

    __local float fi[q], fj[q], fi_t[q], fj_t[q];
    __local float scratch[q*q];

    int i = li%q;
    int j = li/q;

    float ff = (bimarg[n] + pc)/(1.0 + q*q*pc);
    get_unimarg(ff, li, fi, fj, scratch);
    ////apply pseudocount to model
    //ff = (1-u)*(1-u)*ff + ((1-u)*u/q)*(fi[i] + fj[j]) + u*u/(q*q);
    //fi[i] = (1-u)*fi[i] + u/q;
    //fj[j] = (1-u)*fj[j] + u/q;

    float ff_t = (bimarg_target[n] + pc)/(1.0 + q*q*pc);
    get_unimarg(ff_t, li, fi_t, fj_t, scratch);
    ////apply pseudocount to target
    //ff_t = (1-u)*(1-u)*ff_t + ((1-u)*u/q)*(fi_t[i] + fj_t[j]) + u*u/(q*q);
    //fi_t[i] = (1-u)*fi_t[i] + u/q;
    //fj_t[j] = (1-u)*fj_t[j] + u/q;

    float step = -(ff_t - ff)/ff + (((float)L-2)/(L-1))*( 
                           (fi_t[i] - fi[i])/fi[i] + (fj_t[j] - fj[j])/fj[j]  );
    //float step = -(ff_t - ff)/ff;
    //float step = -(bimarg_target[n] - bimarg[n])/(bimarg[n] + u);

    Jo[n] = Ji[n] + gamma*step;
}

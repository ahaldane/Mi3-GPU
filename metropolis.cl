#include <mwc64x/cl/mwc64x/mwc64xvec2_rng.cl>

#ifdef cl_khr_byte_addressable_store
#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
#else
#error "The cl_khr_byte_addresssable_store extension is required!!"
#endif

#pragma OPENCL EXTENSION cl_khr_fp64: enable

typedef unsigned char uchar;
typedef unsigned int uint;

float uniformMap(uint i){
    return (i>>8)*0x1.0p-24f; //converts a 32 bit integer to a float [0,1) 
}

#define NCOUPLE ((L*(L-1)*nB*nB)/2)

#define getbyte(mem, n) (((uchar*)(mem))[n])
#define setbyte(mem, n, val) {(((uchar*)(mem))[n]) = val;}

//expands couplings stored in a (nPair x nB*nB) form to an (L*L x nB*nB) form
__kernel //to be called with group size nB*nB, with nPair groups
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

    __local float lv[nB*nB];
    lv[li] = v[gi*nB*nB + li];
    barrier(CLK_LOCAL_MEM_FENCE);
    vp[nB*nB*(L*i+j) + li] = lv[li];
    vp[nB*nB*(i+L*j) + li] = lv[nB*(li%nB) + li/nB]; 
}

__kernel 
void storeSeqs(__global uint *smallbuf, 
               __global uint *largebuf,
                        uint  nlargebuf,
                        uint  offset){
    uint w, n;
    uint nseqs = get_global_size(0);

    #define SWORDS ((L-1)/4+1) 
    for(w = 0; w < SWORDS; w++){
        for(n = get_local_id(0); n < nseqs; n += get_local_size(0)){
            largebuf[w*nlargebuf + offset + n] = smallbuf[w*nseqs + n];
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
        uint sbn = seqmem[(n/4)*get_global_size(0) + get_global_id(0)]; 
        #pragma unroll //probably ignored
        for(cn = n%4; cn < 4 && n < L-1; cn++, n++){
            seqn = getbyte(&sbn, cn);
            m = n+1;
            while(m < L){
                uint sbm = seqmem[(m/4)*get_global_size(0) + get_global_id(0)]; 

                #pragma unroll
                for(cm = m%4; cm < 4 && m < L; cm++, m++){
                    // could add skip for fixpos here...
                    for(k = li; k < nB*nB; k += get_local_size(0)){
                        lcouplings[k] = J[(n*L + m)*nB*nB + k]; 
                    }
                    barrier(CLK_LOCAL_MEM_FENCE);

                    seqm = getbyte(&sbm, cm);
                    energy = energy + lcouplings[nB*seqn + seqm];
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
                 __global float *energies){
    __local float lcouplings[4*nB*nB];
    energies[get_global_id(0)] = getEnergiesf(J, seqmem, lcouplings);
}

__kernel
void initRNG(__global mwc64xvec2_state_t *rngstates,
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
        for(n = get_local_id(0); n < min((uint)4, L-m)*nB*nB; 
                                                      n += get_local_size(0)){
            lcouplings[n] = J[(pos*L + m)*nB*nB + n]; 
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
            energy += lcouplings[nB*nB*n + nB*mutres + seqm];
            energy -= lcouplings[nB*nB*n + nB*seqp   + seqm];
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
                __global uint *seqmem){
    
    uint nseqs = get_global_size(0);
	mwc64xvec2_state_t rstate = rngstates[get_global_id(0)];

    //set up local mem 
    __local float lcouplings[nB*nB*4];
    __local uint shared_position;

    // The rest of this function is complicated by optimizations for the GPU.
    // For clarity, here is equivalent but clearer (pseudo)code:
    //
    //energy = energies[get_global_id(0)];
    //uint pos = 0; //position to mutate
    //for(i = 0; i < nsteps; i++){
    //    uint pos = rand()%L; //position to mutate
    //    uint residue = rand()%nB; //calculate new residue
    //    float newenergy = energy; //calculate new energy
    //    for(m = 0; m < L; m++){
    //        if(m == pos) continue;
    //        newenergy += (J[pos,m,residue,seq[m]] - J[pos,m,seq[pos],seq[m]]);
    //    }
    //    //apply MC criterion and possibly update
    //    if(exp(-(newenergy - energy)) > rand()){ 
    //        seq[pos] = residue;
    //        energy = newenergy;
    //    }
    //}

    //initialize energy
    float energy = getEnergiesf(J, seqmem, lcouplings);

    uint i;
    for(i = 0; i < nsteps; i++){
        uint pos = position_list[i];
        uint2 rng = MWC64XVEC2_NextUint2(&rstate);
        uchar mutres = rng.x%nB;  // small error here if MAX_INT%nB != 0
                                  // of order nB/MAX_INT in marginals
        uint sbn = seqmem[(pos/4)*nseqs + get_global_id(0)]; 
        uchar seqp = getbyte(&sbn, pos%4); 

        float newenergy = UpdateEnergy(lcouplings, J, seqmem, nseqs, 
                                       pos, seqp, mutres, energy);

        //apply MC criterion and possibly update
        if(exp(-(newenergy - energy)) > uniformMap(rng.y)){ 
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

__kernel //call with group size = NHIST, for nPair groups
void countBimarg(__global uint *bicount,
                 __global float *bimarg, 
                          uint nseq,
                 __global uint *seqmem,
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

    for(n = 0; n < nB*nB; n++){
        hist[nhist*n + li] = 0; 
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    uint tmp;
    //loop through all sequences
    for(n = li; n < nseq; n += nhist){
        tmp = seqmem[(i/4)*nseq+n];
        uchar seqi = ((uchar*)&tmp)[i%4];
        tmp = seqmem[(j/4)*nseq+n];
        uchar seqj = ((uchar*)&tmp)[j%4];
        hist[nhist*(nB*seqi + seqj) + li]++;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    //merge histograms
    for(n = 0; n < nB*nB; n++){
        for(m = nhist/2; m > 0; m >>= 1){
            if(li < m){
                hist[nhist*n + li] = hist[nhist*n + li] + hist[nhist*n + li +m];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }
    
    for(n = li; n < nB*nB; n += nhist){ //only loops once if nhist > nB*nB
        uint count = hist[nhist*n];
        bicount[gi*nB*nB + n] = count;
        bimarg[gi*nB*nB + n] = ((float)count)/nseq;
    }
}

__kernel //call with global work size = to # sequences
void perturbedWeights(__global float *J, 
                      __global uint *seqmem,
                      __global float *weights,
                      __global float *energies){
    __local float lcouplings[4*nB*nB];
    float energy = getEnergiesf(J, seqmem, lcouplings);
    weights[get_global_id(0)] = exp(-(energy - energies[get_global_id(0)]));
}

__kernel //simply sums a vector. Call with single group of size VSIZE
void sumWeights(__global float *weights,
                __global float *sumweights,
                          uint  nseq,
                __local   float *sums){
    uint li = get_local_id(0);
    uint vsize = get_local_size(0);
    uint n;

    sums[li] = 0;
    for(n = li; n < nseq; n += vsize){
        sums[li] += weights[n];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    //reduce
    for(n = vsize/2; n > 0; n >>= 1){
        if(li < n){
            sums[li] = sums[li] + sums[li + n];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if(li == 0){
        *sumweights = sums[0];
    }
}

__kernel //call with group size = NHIST, for nPair groups
void weightedMarg(__global float *bimarg_new, 
                  __global float *weights, 
                  __global float *sumweights,
                           uint nseq, 
                  __global uint *seqmem,
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

    for(n = 0; n < nB*nB; n++){
        hist[nhist*n + li] = 0; 
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    uint tmp;
    //loop through all sequences
    for(n = li; n < nseq; n += nhist){
        tmp = seqmem[(i/4)*nseq+n];
        uchar seqi = ((uchar*)&tmp)[i%4];
        tmp = seqmem[(j/4)*nseq+n];
        uchar seqj = ((uchar*)&tmp)[j%4];
        hist[nhist*(nB*seqi + seqj) + li] += weights[n];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    //merge histograms
    for(n = 0; n < nB*nB; n++){
        for(m = nhist/2; m > 0; m >>= 1){
            if(li < m){
                hist[nhist*n + li] = hist[nhist*n + li] + hist[nhist*n + li +m];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }

    for(n = li; n < nB*nB; n += nhist){ //only loops once if nhist >= nB*nB
        bimarg_new[gi*nB*nB + n] = hist[nhist*n]/(*sumweights);
    }
}

__kernel 
void updatedJ(__global float *bimarg_target,
              __global float *bimarg,
                       float gamma,
                       float pc,
              __global float *J_orig,
                       float jclamp,
              __global float *Ji,
              __global float *Jo){
    uint n = get_global_id(0);

    if(n > NCOUPLE){
        return;
    }

    Jo[n] = Ji[n] - gamma*(bimarg_target[n] - bimarg[n])/(bimarg[n] + pc);

    if(jclamp != 0){
        Jo[n] = clamp(Jo[n], J_orig[n] - jclamp, 
                             J_orig[n] + jclamp);
    }
}

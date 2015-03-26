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
               __global uint *offset_p){
    uint w, n, offset;
    offset = *offset_p;

    #define SWORDS ((L-1)/4+1) 
    for(w = 0; w < SWORDS; w++){
        for(n = get_local_id(0); n < NSEQS; n += WGSIZE){
            largebuf[w*NSAMPLES*NSEQS + offset + n] = smallbuf[w*NSEQS + n];
        }
    }
    #undef SWORDS
}

//only call from kernels with NSEQ work units!!!!!!
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
            seqn = ((uchar*)(&sbn))[cn];
            m = n+1;
            while(m < L){
                uint sbm = seqmem[(m/4)*get_global_size(0) + get_global_id(0)]; 
                
                ////temporary code which does not work
                //uint moff = m%4; //offset within the byte where we start
                ////pre-load next 4 rows of couplings (minus offset)
                //for(k = get_local_id(0) + (nB*nB*moff); 
                //    k < min((uint)4, L-m)*nB*nB; 
                //    k += get_local_size(0)){
                //    lcouplings[k] = J[(n*L + m-moff)*nB*nB + k]; 
                //}
                //barrier(CLK_LOCAL_MEM_FENCE);
                
                ////calculate contribution of next 4 letters
                //#pragma unroll
                //for(cm = moff; cm < 4 && m < L; cm++, m++){
                //    seqm = ((uchar*)(&sbm))[cm];
                //    energy += lcouplings[nB*nB*cm + nB*seqn + seqm];
                //}
                //barrier(CLK_LOCAL_MEM_FENCE);

                #pragma unroll
                for(cm = m%4; cm < 4 && m < L; cm++, m++){
                    for(k = li; k < nB*nB; k += get_local_size(0)){
                        lcouplings[k] = J[(n*L + m)*nB*nB + k]; 
                    }
                    barrier(CLK_LOCAL_MEM_FENCE);

                    seqm = ((uchar*)(&sbm))[cm];
                    energy = energy+lcouplings[nB*seqn + seqm];
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

//#define getbyte(mem, n) (((mem>>(8*n))&0xff))
//#define setbyte(mem, n, val) {mem = mem ^ ((val ^ getbyte(mem,n))<<(8*n));}

//uses fewer registers
#define getbyte(mem, n) (((uchar*)(&mem))[n])
#define setbyte(mem, n, val) {(((uchar*)(&mem))[n]) = val;}

__kernel //__attribute__((work_group_size_hint(WGSIZE, 1, 1)))
void metropolis(__global float *J,
                __global uint *run_seed, 
                __global uint *gpu_seed, 
                __global uint *nsteps_p, 
                __global float *energies,
                __global uint *seqmem){
    
    *run_seed += 1;
    uint nsteps = *nsteps_p;

    //init rng
	mwc64xvec2_state_t rstate = {(uint2)(get_local_id(0), get_global_id(0)),
                                 (uint2)(*run_seed, *gpu_seed)};
    MWC64XVEC2_NextUint2(&rstate); //step once past seed

    //set up local mem 
    __local float lcouplings[nB*nB*4];

    // The rest of this function is complicated by optimizations for the GPU.
    // For clarity, here is equivalent but clearer (pseudo)code:
    //
    //energy = energies[get_global_id(0)];
    //uint pos = 0; //position to mutate
    //for(i = 0; i < nsteps; i++){
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
    //    pos = (pos+1)%L; //pos cycles repeatedly through all positions
    //}

    //initialize energy
    float energy = getEnergiesf(J, seqmem, lcouplings);

    //main loop
    uint pos = 0;
    uint sbn;
    uint i;
    for(i = 0; i < nsteps; i++){
        uint2 rng = MWC64XVEC2_NextUint2(&rstate);
        uint mutres = rng.x%nB; //small error here if MAX_INT%nB != 0
        
        //calculate new energy
        float newenergy = energy; 
        if(pos%4 == 0){ //load next 4 bytes of sequence
            sbn = seqmem[(pos/4)*NSEQS + get_global_id(0)]; 
        }
        uchar seqp = getbyte(sbn, pos%4);
        uint m = 0;
        while(m < L){
            //loop through seq, changing energy by changed coupling with pos
            
            //load the next 4 rows of couplings to local mem
            uint n;
            for(n = get_local_id(0); n < min((uint)4, L-m)*nB*nB; n += WGSIZE){
                lcouplings[n] = J[(pos*L + m)*nB*nB + n]; 
            }

            uint sbm;
            if(m/4 == pos/4){
                //careful that sbn is not stored back to seqmem for 4 steps
                sbm = sbn;
            }
            else{
                //bottleneck of this kernel
                sbm = seqmem[(m/4)*NSEQS + get_global_id(0)]; 
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            
            //calculate contribution of those 4 rows to energy
            for(n = 0; n < 4 && m < L; n++, m++){
                if(m == pos){
                    continue;
                }
                uchar seqm = getbyte(sbm, n);
                newenergy = newenergy + lcouplings[nB*nB*n + nB*mutres + seqm];
                newenergy = newenergy - lcouplings[nB*nB*n + nB*seqp   + seqm];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        //apply MC criterion and possibly update
        if(exp(-(newenergy - energy)) > uniformMap(rng.y)){ 
            setbyte(sbn, pos%4, mutres);
            energy = newenergy;
        }

        if(((pos+1)%4 == 0) || (pos+1 == L)){ 
            //store the finished 4 bytes of sequence
            seqmem[(pos/4)*NSEQS + get_global_id(0)] = sbn;
        }
        
        //pos cycles repeatedly through all positions, in order.
        //this helps the sequence & J loads to be coalesced
        pos = (pos+1)%L;
    }

#ifdef MEASURE_FP_ERROR
    energies[get_global_id(0)] = energy;
#endif
}

__kernel //call with group size = NHIST, for nPair groups
void countBimarg(__global uint *bicount,
                 __global float *bimarg, 
                 __global uint *nseq_p, 
                 __global uint *seqmem) {
    uint li = get_local_id(0);
    uint gi = get_group_id(0);
    uint i,j,n,m;
    uint nseq = *nseq_p;
    
    //figure out which i,j pair we are
    i = 0;
    j = L-1;
    while(j <= gi){
        i++;
        j += L-1-i;
    }
    j = gi + L - j; //careful with underflow!

    __local uint hist[nB*nB*NHIST];
    for(n = 0; n < nB*nB; n++){
        hist[NHIST*n + li] = 0; 
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    uint tmp;
    //loop through all sequences
    for(n = li; n < nseq; n += NHIST){
        tmp = seqmem[(i/4)*nseq+n];
        uchar seqi = ((uchar*)&tmp)[i%4];
        tmp = seqmem[(j/4)*nseq+n];
        uchar seqj = ((uchar*)&tmp)[j%4];
        hist[NHIST*(nB*seqi + seqj) + li]++;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    //merge histograms
    for(n = 0; n < nB*nB; n++){
        for(m = NHIST/2; m > 0; m >>= 1){
            if(li < m){
                hist[NHIST*n + li] = hist[NHIST*n + li] + hist[NHIST*n + li +m];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }
    
    for(n = li; n < nB*nB; n += NHIST){ //only loops once if NHIST > nB*nB
        uint count = hist[NHIST*n];
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
                __global uint *nseq_p){
    uint li = get_local_id(0);
    uint n;
    uint nseq = *nseq_p;

    __local float sums[VSIZE];
    sums[li] = 0;
    for(n = li; n < nseq; n += VSIZE){
        sums[li] += weights[n];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    //reduce
    for(n = VSIZE/2; n > 0; n >>= 1){
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
                  __global uint *nseq_p, 
                  __global uint *seqmem) {
    uint li = get_local_id(0);
    uint gi = get_group_id(0);
    uint i,j,n,m;
    uint nseq = *nseq_p;
    
    //figure out which i,j pair we are
    i = 0;
    j = L-1;
    while(j <= gi){
        i++;
        j += L-1-i;
    }
    j = gi + L - j; //careful with underflow!

    __local float hist[nB*nB*NHIST];
    for(n = 0; n < nB*nB; n++){
        hist[NHIST*n + li] = 0; 
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    uint tmp;
    //loop through all sequences
    for(n = li; n < nseq; n += NHIST){
        tmp = seqmem[(i/4)*nseq+n];
        uchar seqi = ((uchar*)&tmp)[i%4];
        tmp = seqmem[(j/4)*nseq+n];
        uchar seqj = ((uchar*)&tmp)[j%4];
        hist[NHIST*(nB*seqi + seqj) + li] += weights[n];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    //merge histograms
    for(n = 0; n < nB*nB; n++){
        for(m = NHIST/2; m > 0; m >>= 1){
            if(li < m){
                hist[NHIST*n + li] = hist[NHIST*n + li] + hist[NHIST*n + li +m];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }

    for(n = li; n < nB*nB; n += NHIST){ //only loops once if NHIST >= nB*nB
        bimarg_new[gi*nB*nB + n] = hist[NHIST*n]/(*sumweights);
    }
}

__kernel 
void updatedJ(__global float *bimarg_target,
              __global float *bimarg,
              __global float *J_orig,
              __global float *gamma,
              __global float *Ji,
              __global float *Jo){
    uint n = get_global_id(0);

    if(n > NCOUPLE){
        return;
    }

    Jo[n] = Ji[n] - (*gamma)*(bimarg_target[n] - bimarg[n])/(bimarg[n] + PC);

    #ifdef JCUTOFF
    Jo[n] = clamp(Jo[n], J_orig[n] - (float)JCUTOFF, 
                         J_orig[n] + (float)JCUTOFF);
    #endif
}


#include <Random123/threefry.h>

//#ifdef cl_nv_pragma_unroll
//#pragma OPENCL EXTENSION cl_nv_pragma_unroll : enable
//#endif

#ifdef cl_khr_byte_addressable_store
#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
#else
#error "The cl_khr_byte_addresssable_store extension is required!!"
#endif

#pragma OPENCL EXTENSION cl_khr_fp64: enable

typedef unsigned char uchar;
typedef unsigned int uint;

float uniformMap(uint i){
    return (i>>8)*0x1.0p-24f; //converts a 32 bit integer to a uniform float [0,1) 
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

__kernel //to be called with group size nB*nB, with nPair groups
void genLogScoreSeqs(__global float *unimarg, 
                     __global uint *seqmem){
    uint li = get_local_id(0);
    uint gi = get_group_id(0);
}


inline double getEnergiesf(__global float *J,
                           __global uint *seqmem,
                           __global float *energies,
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
    double energy = 0;
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
                    energy += lcouplings[nB*seqn + seqm];
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
    energies[get_global_id(0)] = getEnergiesf(J, seqmem, energies, lcouplings);
}

__kernel //__attribute__((work_group_size_hint(WGSIZE, 1, 1)))
void metropolis(__global float *J,
                __global uint *run_seed, 
                __global uint *seqmem,
                __global float *energies){
    uint i,n,m;
    uint li = get_local_id(0);
    uint gi = get_global_id(0);

    //init rng
    threefry4x32_key_t k = {{0, get_local_id(0), *run_seed, SEED}};
    threefry4x32_ctr_t c = {{0, get_global_id(0), 0xdeadbeef, 0xbeeff00d}};
    
    //set up local mem 
    uchar seqm, seqp;
    uint cm;
    uint sbn, sbm;
    __local float lcouplings[nB*nB*4];

    // The rest of this function is complicated by optimizations for the GPU.
    // For clarity, here is equivalent but clearer (pseudo)code:
    //
    //energy = getEnergies();
    //uint pos = 0; //position to mutate
    //for(i = 0; i < nsteps; i++){
    //    uint residue = rand()%nB; //calculate new residue
    //    double newenergy = energy; //calculate new energy
    //    for(m = 0; m < L; m++){
    //        newenergy += (J[pos,m,residue,seq[m]] - J[pos,m,seq[pos],seq[m]]);
    //    }
    //    //apply MC criterion and possibly update
    //    if(exp(-(newenergy - energy)) > rand()){ 
    //        seq[pos] = residue;
    //        energy = newenergy;
    //    }
    //    pos = (pos+1)%L; //pos cycles repeatedly through all positions
    //}
    //energies[get_global_id(0)] = energy;

    //initialize energy
    double energy = getEnergiesf(J, seqmem, energies, lcouplings);

    //main loop
    uint pos = 0;
    for(i = 0; i < nsteps; i++){
        c.v[0]++; //we only use 2 of the 4 threefry values...
        threefry4x32_ctr_t rng = threefry4x32(c, k); 
        uint residue = rng.v[0]%nB; 
        
        //calculate new energy
        double newenergy = energy;
        uint sbn = seqmem[(pos/4)*NGROUPS*WGSIZE + gi]; 
        seqp = ((uchar*)(&sbn))[pos%4];
        m = 0;
        while(m < L){
            //loop through seq, changing energy by changed coupling with pos
            uint sbm = seqmem[(m/4)*NGROUPS*WGSIZE + gi]; 
            
            //load the next 4 rows of couplings to local mem
            for(n = li; n < min((uint)4, L-m)*nB*nB; n += WGSIZE){
                lcouplings[n] = J[(pos*L + m)*nB*nB + n]; 
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            
            //calculate contribution of those 4 rows to energy
            #pragma unroll
            for(cm = 0; cm < 4 && m < L; cm++, m++){
                seqm = ((uchar*)(&sbm))[cm];
                newenergy += ( lcouplings[nB*nB*cm + nB*residue + seqm] - 
                               lcouplings[nB*nB*cm + nB*seqp    + seqm] );
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        //apply MC criterion and possibly update
        float p = exp(-(newenergy - energy));
        if(p > uniformMap(rng.v[1]) && ! isinf(newenergy)){ 
        //no need to check for nan:  the old coupling is guaranteed not inf
            ((uchar*)(&sbn))[pos%4] = residue;
            seqmem[(pos/4)*NGROUPS*WGSIZE + gi] = sbn;
            energy = newenergy;
        }
        
        //pos cycles repeatedly through all positions, in order
        //this helps the sequence & J loads to be coalesced
        pos = (pos+1)%L;
    }

    energies[gi] = energy;
}

__kernel //call with group size = NHIST, for nPair groups
void countBimarg(__global uint *bicount,
                 __global float *bimarg, 
                 __global uint *offset,
                 __global uint *nseq_p, 
                 __global uint *seqmem) {
    uint li = get_local_id(0);
    uint gi = get_group_id(0) + (*offset);
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
                     __global uint *nseq_p, 
                      __global uint *seqmem,
                      __global float *weights,
                      __global float *energies){
    __local float lcouplings[4*nB*nB];
    double energy = getEnergiesf(J, seqmem, energies, lcouplings);
    weights[get_global_id(0)] = exp(-(energy - energies[get_global_id(0)]));
}

__kernel //simply sums a vector. Call with single group as large as possible
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
                  __global uint *offset,
                  __global uint *nseq_p, 
                  __global uint *seqmem) {
    uint li = get_local_id(0);
    uint gi = get_group_id(0) + (*offset);
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
              __global float *alpha,
              __global float *Ji,
              __global float *Jo){
    uint li = get_local_id(0);
    uint n;
    for(n = li; n < NCOUPLE; n += VSIZE){
        Jo[n] = Ji[n] - (*alpha)*(bimarg_target[n] - bimarg[n])/(bimarg[n] + PC);

        #ifdef JCUTOFF
        Jo[n] = clamp(Jo[n], J_orig[n] - (float)JCUTOFF, J_orig[n] + (float)JCUTOFF);
        #endif
    }
}


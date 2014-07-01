#include <Random123/threefry.h>

//#ifdef cl_nv_pragma_unroll
//#pragma OPENCL EXTENSION cl_nv_pragma_unroll : enable
//#endif

#ifdef cl_khr_byte_addressable_store
#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
#else
#error "The cl_khr_byte_addresssable_store extension is required!!"
#endif

typedef uchar uchar;
typedef uint uint;

float uniformMap(uint i){
    return (i>>8)*0x1.0p-24f; //converts a 32 bit integer to a uniform float [0,1) 
}

#define NCOUPLE ((L*(L-1)*nB*nB)/2)

//sequences are padded to 32 bit boundaries
#define SWORDS ((L-1)/4+1)  //number of words needed to store a sequence
#define SBYTES (4*SWORDS)   //number of bytes needed to store a sequence

//kernel which fills in seqmem with the starting sequence
__kernel //__attribute__((work_group_size_hint(WGSIZE, 1, 1)))
void initSeqMem(__global uint *startseq, __global uint *savedSeqs){
    uint i;
    for(i = 0; i < SWORDS; i++){
        savedSeqs[i*NGROUPS*WGSIZE + get_global_id(0)] = startseq[i];
    }
}

//kernel which zeros bicounts. (in OpenCL 1.2+, clEnqueueFillBuffer does this)
__kernel
void zeroBicounts(__global uint *bicounts){
    if(get_global_id(0) < NCOUPLE)
        bicounts[get_global_id(0)] = 0;
}

//convenient macro to access J matrix elements:
#define coupling(i,j,a,b) J[(b) + nB*(a) + nB*nB*((i)*L-(i)*((i)+1)/2 + (j)-(i)-1)]
//assumes j > i!!

//kernel which iterates the MCMC sequence generation steps, and stores
//final sequences in savedSeqMem
__kernel //__attribute__((work_group_size_hint(WGSIZE, 1, 1)))
void metropolis(__global float *J,
                __global uint *run_seed, 
                __global uint *savedSeqMem){
    uint i,j,n,m;

    //init rng
    threefry4x32_key_t k = {{0, get_local_id(0), *run_seed, SEED}};
    threefry4x32_ctr_t c = {{0, get_global_id(0), 0xdeadbeef, 0xbeeff00d}};
    
    //set up local mem 
    uint seqmem[SWORDS]; //32bit aligned
    uchar *seq = (uchar*)seqmem;
    float energy = 0;
    
    //initialize sequence
    for(i = 0; i < SWORDS; i++){
        seqmem[i] = savedSeqMem[i*NGROUPS*WGSIZE + get_global_id(0)];
    }
    for(n = 0; n < L-1; n++){
        for(m = n+1; m < L; m++){
            energy += coupling(n,m,seq[n],seq[m]);
        }
    }

    //main loop
    for(i = 0; i < nsteps; i++){
        c.v[0]++; 
        threefry4x32_ctr_t rng = threefry4x32(c, k);
        
        //repeat sequence update 2x, using 2 of the 4 rngs each time
        #pragma unroll
        for(j = 0; j <= 2; j += 2){
            uint r = rng.v[j]%(nB*L);
            uint pos = r/nB;
            uint residue = r%nB; 

            float newenergy = energy;
            for(m = 0; m < L; m++){
                if(m < pos){
                    newenergy += ( coupling(m,pos,seq[m],residue) - 
                                   coupling(m,pos,seq[m],seq[pos]) );
                }
                if(m > pos){
                    newenergy += (coupling(pos,m,residue,seq[m]) - 
                                  coupling(pos,m,seq[pos],seq[m]) );
                }
            }

            float p = exp(-(newenergy - energy));
            if(p > uniformMap(rng.v[j+1]) && ! isinf(newenergy)){ 
            //no need to check for nan:  the old coupling is guaranteed not inf
                seq[pos] = residue;
                energy = newenergy;
            }
        }
    }
    
    //store final sequence
    for(i = 0; i < SWORDS; i++){
        savedSeqMem[i*NGROUPS*WGSIZE + get_global_id(0)] = seqmem[i];
    }
}

//kernel which computes bicounts from savedSeqMem
__kernel //__attribute__((work_group_size_hint(WGSIZE, 1, 1)))
void countSeqs(__global uint *bicounts, 
               __global uint *savedSeqs, 
               __global uint *pairI, 
               __global uint *pairJ) {
    __local uint smem[WGSIZE*SWORDS];
    __local uchar *seqs = (__local uchar*)smem;

    uint li = get_local_id(0);
    uint n = get_global_id(0);

    uint pairnum, bases, s1, s2;
    uchar b1, b2;
    uint count = 0;
    uint i,s,m;

    if(n < NCOUPLE){
        pairnum = n/(nB*nB);
        bases = n%(nB*nB);
        s1 = pairI[pairnum]; 
        s2 =  pairJ[pairnum]; 
        b1 = bases/nB;
        b2 = bases%nB;
    }
    //Note: in case pairI and pairJ do not fit in memory, could use:
    // s1 = L-1-(int)((1+sqrt(1+8*(L*(L-1)/2-1 - pairnum)))/2); 
    // s2 = pairnum+s1+1-s1*L+s1*(s1)/2

    for(s = 0; s < NGROUPS; s++){
        //load some sequences into local memory
        for(i = 0; i < SWORDS; i++){
            smem[li*SWORDS + i] = savedSeqs[i*NGROUPS*WGSIZE + (s*WGSIZE + li)];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
       
        //count them up
        if(n < NCOUPLE){
            for(m = 0; m < WGSIZE; m++){
                //this probably causes lots of bank conflicts
                if(seqs[SBYTES*m + s1] == b1  && seqs[SBYTES*m + s2] == b2){ 
                    count++;
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if(n < NCOUPLE){
        bicounts[n] += count; 
    }
}

//kernel which updates the couplings based on the measured bicounts
__kernel 
void updateCouplings(__global float *targetMarginals, 
                     __global uint *bicounts, 
                     __global float *couplings){
    uint li = get_local_id(0);
    uint n = get_global_id(0);

    if(n >= NCOUPLE){
        return;
    }

    float marginal = ((float)bicounts[n])/((float)nseqs);
    float target = targetMarginals[n];

    if(target == 0){
        couplings[n] = INFINITY;
    }
    else{
        couplings[n] += gamma*(target - marginal)/(target*(target-1));
    }
}


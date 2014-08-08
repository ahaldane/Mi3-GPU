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

//sequences are padded to 32 bit boundaries
#define SWORDS ((L-1)/4+1)  //number of words needed to store a sequence
#define SBYTES (4*SWORDS)   //number of bytes needed to store a sequence

//convenient macro to access J matrix elements:
#define IND(i,j,a,b) ((b) + nB*(a) + nB*nB*((i)*L-(i)*((i)+1)/2 + (j)-(i)-1))
//assumes j > i!!

__kernel //__attribute__((work_group_size_hint(WGSIZE, 1, 1)))
void metropolis(__global float *J,
                __global uint *run_seed, 
                __global uint *seqmem,
                __global float *energies){
    uint i,j,n,m;

    //init rng
    threefry4x32_key_t k = {{0, get_local_id(0), *run_seed, SEED}};
    threefry4x32_ctr_t c = {{0, get_global_id(0), 0xdeadbeef, 0xbeeff00d}};
    
    //set up local mem 
    uchar seqm, seqn, seqp;
    uint cn, cm;
    uint sbn, sbm;
    __local float lcouplings[nB*nB];

    //initialize energy
    double energy = 0;
    n = 0;
    while(n < L-1){
        // coalsesced load
        uint sbn = seqmem[(n/4)*NGROUPS*WGSIZE + get_global_id(0)]; 
        #pragma unroll
        for(cn = n%4; cn < 4 && n < L-1; cn++, n++){
            seqn = ((uchar*)(&sbn))[cn];
            m = n+1;
            while(m < L){

                uint sbm = seqmem[(m/4)*NGROUPS*WGSIZE + get_global_id(0)]; 
                #pragma unroll
                for(cm = m%4; cm < 4 && m < L; cm++, m++){
                    if(get_local_id(0) < nB*nB){
                        //WGSIZE must be greater than nB*nB!!
                        lcouplings[get_local_id(0)] = J[(n*L + m)*nB*nB + get_local_id(0)];
                    }
                    barrier(CLK_LOCAL_MEM_FENCE);

                    seqm = ((uchar*)(&sbm))[cm];
                    energy += lcouplings[nB*seqn + seqm];
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
            }
        }
    }

    //with some super preprocessor magic, could do unrolling myself?

    //main loop
    uint pos = 0;
    for(i = 0; i < nsteps; i++){
        c.v[0]++; 
        threefry4x32_ctr_t rng = threefry4x32(c, k);
        
        //repeat sequence update 2x, using 2 of the 4 rngs each time
        #pragma unroll
        for(j = 0; j <= 1; j += 2){
            uint residue = rng.v[j]%nB; 
            
            //calculate new energy
            double newenergy = energy;
            uint sbn = seqmem[(pos/4)*NGROUPS*WGSIZE + get_global_id(0)]; 
            seqp = ((uchar*)(&sbn))[pos%4];
            m = 0;
            while(m < L){
                uint sbm = seqmem[(m/4)*NGROUPS*WGSIZE + get_global_id(0)]; 
                #pragma unroll
                for(cm = 0; cm < 4 && m < L; cm++, m++){
                    if(get_local_id(0) < nB*nB){
                        lcouplings[get_local_id(0)] = J[(pos*L + m)*nB*nB + get_local_id(0)]; 
                    }
                    barrier(CLK_LOCAL_MEM_FENCE);
                    seqm = ((uchar*)(&sbm))[cm];
                    newenergy += ( lcouplings[nB*residue + seqm] - 
                                   lcouplings[nB*seqp    + seqm] );
                    barrier(CLK_LOCAL_MEM_FENCE);
                 }
            }

            //apply MC criterion and possibly update
            float p = exp(-(newenergy - energy));
            if(p > uniformMap(rng.v[j+1]) && ! isinf(newenergy)){ 
            //no need to check for nan:  the old coupling is guaranteed not inf
                ((uchar*)(&sbn))[pos%4] = residue;
                seqmem[(pos/4)*NGROUPS*WGSIZE + get_global_id(0)] = sbn;
                energy = newenergy;
            }
            
            //pos cycles repeatedly through all positions, in order
            //this helps the sequence & J loads to be coalesced
            pos = (pos+1)%L;
        }
    }

    energies[get_global_id(0)] = energy;
}

//kernel which computes bicounts from savedSeqMem
__kernel //__attribute__((work_group_size_hint(WGSIZE, 1, 1)))
void countSeqs(__global uint *bicounts, 
               __global uint *seqmem, 
               __global uint *pairI, 
               __global uint *pairJ) {
    __local uint smem[NSEQLOAD*SWORDS];
    __local uchar *seqs = (__local uchar*)smem;

    uint li = get_local_id(0);
    uint n = get_global_id(0);
    uint word;

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

    //NSEQLOAD must be a divisor of NGROUPS*WGSIZE!!!  The idea is we divide up 
    //the NGROUPS*WGSIZE sequences into chunks of size NSEQLOAD.
    //Since each work unit loads one sequence, must have NSEQLOAD < WGSIZE
    for(s = 0; s < NGROUPS*WGSIZE; s += NSEQLOAD){
        //load NSEQLOAD sequences into local memory
        if(li < NSEQLOAD){
            for(i = 0; i < SWORDS; i++){
                smem[li*SWORDS + i] = seqmem[i*NGROUPS*WGSIZE + (s + li)];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
       
        //count them up
        if(n < NCOUPLE){
            for(m = 0; m < NSEQLOAD; m++){
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

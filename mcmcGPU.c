#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include <math.h>
#include <string.h>
#include <assert.h>
#include <limits.h>
#include <float.h>
#include <errno.h>
#include "common/epsilons.h"
#include "gpuSetup.h"

#define WGSIZE 256
#define NGROUPS 32

//sequences are padded to 32 bit boundaries
#define SWORDS ((L-1)/4+1)  //number of words needed to store a sequence
#define SBYTES (4*SWORDS)   //number of bytes needed to store a sequence

#define  SEQMEMSIZE  SWORDS*NGROUPS*WGSIZE // # of words to store all sequences

typedef unsigned char uchar;
typedef unsigned int uint;
//I also assume uint is the same size as cl_uint, etc


static cl_kernel clMCMCKernel, clCountKernel, clUpdateKernel, clZeroKernel;

static cl_mem J_dev;
static cl_mem bicount_dev;
static cl_mem bimarg_dev;
static cl_mem pairI_dev;
static cl_mem pairJ_dev;
static cl_mem run_seed_dev;
static cl_mem savedseq_dev;

static cl_float *couplings, *bimarg;
static cl_uint *seqmem, *bicount, *pairI, *pairJ;
static uint L, nB, n_couplings, nseqs;

static float gamma_d;
static uint gdsteps, burnstart, burnin, nloop, nsteps, nseqs;

uint readuint(char *str){
    char *end;
    errno = 0;
    long  lnum = strtoul(str, &end, 10);  
    if(end == str || errno){
        fprintf(stderr, "Error reading integer: '%s'\n", str);
        exit(1);
    }
    if((lnum > INT_MAX) || (lnum < INT_MIN))
        fprintf(stderr, "Error: %ld out of range for INT\n", lnum);
    return (int) lnum;
}
float readfloat(char *str){
    char *end;
    errno = 0;
    double  lnum = strtod(str, &end);  
    if(end == str || errno){
        fprintf(stderr, "Error reading float: '%s'\n", str);
        exit(1);
    }
    if((lnum > FLT_MAX) || (lnum < FLT_MIN))
        fprintf(stderr, "Error: %g out of range\n", lnum);
    return (float) lnum;
}

void setupHostFromArgs(int argc, char *argv[]){
    FILE *f;
    uint i;

    int maxargs = 10;
    int argnum = 1;
    if(argc != (maxargs-1) && argc != maxargs){
        printf("Usage: ./a.out bimarg gamma gdsteps initialBurnin burnloop "
               "nloop niter initseqfile [couplings]\n");
        exit(1);
    }
    //XXX I do NO error checking of input parameters after this!!!

    printf("Initialization\n===============\n");
 
    // set seed for rand()
    //srand(2007);
    srand((unsigned)time(NULL));
    
    //read in bimarginals, and determine L and nB
    uint nPairs, nComb; 
    double *bimarg_in = readEM(&nComb, &nPairs, argv[argnum++]);
    L = (int)(((1+sqrt(1+8*nPairs))/2) + 0.5); 
    nB = (int)(sqrt(nComb) + 0.5);//+0.5 for rounding any fp error
    n_couplings = L*(L-1)*nB*nB/2;
    printf("nBases %d  seqLen %d\n", nB, L);
    
                                          // example values
    gamma_d = readfloat(argv[argnum++]);  // 0.005
    gdsteps = readuint(argv[argnum++]);   // 10
    burnstart = readuint(argv[argnum++]); // 100
    burnin = readuint(argv[argnum++]);    // 100
    nloop = readuint(argv[argnum++]);     // 100
    nsteps = readuint(argv[argnum++]);    // 100
    nseqs = WGSIZE*NGROUPS*nloop;

    //malloc host memory
    bicount = malloc(sizeof(cl_uint)*n_couplings);
    couplings = malloc(sizeof(cl_float)*n_couplings);
    bimarg = malloc(sizeof(cl_float)*n_couplings);
    seqmem = malloc(sizeof(cl_uint)*SEQMEMSIZE);
    
    //copy over bimarginals to float array
    for(i = 0; i < n_couplings; i++){
        bimarg[i] = bimarg_in[i];
    }
    free(bimarg_in);
    
    //read in sequences
    uint readval;
    f = fopen(argv[argnum++], "rb");
    #define checkval(param) \
        fread(&readval, sizeof(uint), 1, f); \
        if(readval != param){ \
            printf("Error: Sequence file says " #param "=%d, "\
                   "but running with %d", readval, param); \
            exit(1);  \
        }
    checkval(WGSIZE);
    checkval(NGROUPS);
    checkval(L);
    checkval(nB);
    #undef checkval
    fread(seqmem, sizeof(cl_uint), SEQMEMSIZE, f);
    fclose(f);
    
    //read in couplings, or initialize them to 0
    if(argc == maxargs){
        f = fopen(argv[argnum++], "rb");
        fread(couplings, sizeof(cl_float), n_couplings, f);
        fclose(f);
    }else{
        for(i = 0; i < n_couplings; i++){
            if(bimarg[i] == 0){
                couplings[i] = INFINITY;
            }
            else{
                couplings[i] = 0;
            }
        }
    }
    printf("Initial couplings: ");
    for(i = 0; i < 5; i++){
        printf("%.15g ", couplings[i]);
    }
    printf(" ...\n\n");

    //set up the map from pair index to pair values
    pairI = malloc(sizeof(cl_uint)*L*(L-1)/2);
    pairJ = malloc(sizeof(cl_uint)*L*(L-1)/2);
    uint j,n = 0;
    for(i = 0; i < L-1; i++){
        for(j = i+1; j < L; j++){
            pairI[n] = i;
            pairJ[n] = j;
            n++;
        }
    }
}

void setupGPU(){
    initGPU();

    char options[1024];
    sprintf(options, "-D NGROUPS=%d -D WGSIZE=%d -D SEED=%d -D nB=%d -D L=%d "
                     "-D nsteps=%d -D nseqs=%d -D gamma=%g "
                     "-cl-nv-verbose -Werror", 
                     NGROUPS, WGSIZE, rand(), nB, L, nsteps, nseqs, gamma_d);
    printf("Options: %s\n\n", options);
    loadKernel("metropolis.cl", options);
    //dumpPTX("metropolis.ptx");
    
    J_dev = createRWBufferFilled(sizeof(cl_float)*n_couplings, couplings);
    bicount_dev = createRWBufferEmpty(sizeof(cl_uint)*n_couplings);
    bimarg_dev = createRWBufferFilled(sizeof(float)*n_couplings, bimarg);
    run_seed_dev = createRWBufferEmpty(sizeof(cl_uint));
    savedseq_dev = createRWBufferFilled(sizeof(cl_uint)*SEQMEMSIZE, seqmem);
    pairI_dev = createRWBufferFilled(sizeof(cl_uint)*L*(L-1)/2, pairI);
    pairJ_dev = createRWBufferFilled(sizeof(cl_uint)*L*(L-1)/2, pairJ);

    clMCMCKernel = setupKernel("metropolis", 3, 
                             J_dev, run_seed_dev, savedseq_dev);
    clUpdateKernel = setupKernel("updateCouplings", 3, 
                               bimarg_dev, bicount_dev, J_dev);
    clCountKernel = setupKernel("countSeqs", 4, 
                               bicount_dev, savedseq_dev, pairI_dev, pairJ_dev);
    clZeroKernel = setupKernel("zeroBicounts", 1, bicount_dev);
}

void cleanUp(){
    clReleaseKernel(clMCMCKernel);
    clReleaseKernel(clUpdateKernel);
    clReleaseKernel(clCountKernel);
    clReleaseKernel(clZeroKernel);
    clReleaseMemObject(J_dev);
    clReleaseMemObject(bicount_dev);
    clReleaseMemObject(bimarg_dev);
    clReleaseMemObject(pairI_dev);
    clReleaseMemObject(pairJ_dev);
    clReleaseMemObject(run_seed_dev);
    clReleaseMemObject(savedseq_dev);
    freeGPU();
    free(bicount);
    free(couplings);
    free(bimarg);
    free(seqmem);
    free(pairI);
    free(pairJ);
}

void writeStatus(){
    uint i,j,a;
    uint nPairs = L*(L-1)/2;

    readData(couplings, J_dev, sizeof(uint)*n_couplings);
    readData(bicount, bicount_dev, sizeof(cl_uint)*n_couplings);
    readData(seqmem, savedseq_dev, sizeof(cl_uint)*SEQMEMSIZE);
    
    //print out rmsd
    float rmsd = 0;
    for(j = 0; j < n_couplings; j++){
        float marg = ((float)bicount[j])/((float)nseqs);
        float target = bimarg[j];
        rmsd += (target - marg)*(target - marg);
    }
    printf("RMSD: %g\n", rmsd);
    
    //print some details to stdout
    uint ndisp = 5;
    printf("Bicounts: ");
    for(a = 0; a < ndisp; a++){
        printf("%u ", bicount[a]);
    }
    printf(" ...\nMarginals: ");
    for(a = 0; a < ndisp; a++){
        printf("%g ", ((float)bicount[a])/((float)nseqs));
    }
    printf(" ...\nCouplings: ");
    for(a = 0; a < ndisp; a++){
        printf("%.15g ", couplings[a]);
    }
    printf(" ...\n");
    
    //save current state to file
    FILE *f = fopen("result", "wt");
    for(j = 0; j < nPairs; j++){
        for(a = 0; a < nB*nB; a++){
            fprintf(f, "%.15g ", couplings[nB*nB*j + a]);
        }
        fprintf(f, "\n");
    }
    fclose(f);

    f = fopen("binresult", "wb");
    fwrite(couplings, sizeof(cl_float), n_couplings, f);
    fclose(f);

    f = fopen("bicounts", "wt");
    for(j = 0; j < nPairs; j++){
        for(a = 0; a < nB*nB; a++){
            fprintf(f, "%u ", bicount[nB*nB*j + a]);
        }
        fprintf(f, "\n");
    }
    fclose(f);

    uint wgsize = WGSIZE;
    uint ngroups = NGROUPS;
    f = fopen("finalseqs", "wb");
    fwrite(&wgsize, sizeof(uint), 1, f);
    fwrite(&ngroups, sizeof(uint), 1, f);
    fwrite(&L, sizeof(uint), 1, f);
    fwrite(&nB, sizeof(uint), 1, f);
    fwrite(seqmem, sizeof(cl_uint), SEQMEMSIZE, f);
    fclose(f);
}

int main(int argc, char* argv[]){
    int i,j,l,n,a,b;
    
    //initialize memory, GPU, kernels, buffers
    setupHostFromArgs(argc, argv);
    setupGPU();

    //global size of kernels that run over n_couplings
    //is minimal multiple of wgsize greater than ncoupling
    cl_uint cplgrp = ((n_couplings-1)/WGSIZE+1);
    
    //perform the computation!
    printf("\n\nMCMC Run\n========\n");
    
    //initial burnin
    uint kernel_seed = 0;
    printf("Initial Burnin for %d steps:\n", burnstart);
    for(j = 0; j < burnstart; j++){
        writeData(run_seed_dev, &kernel_seed, sizeof(cl_uint));
        kernel_seed++;
        runKernel(clMCMCKernel, WGSIZE, WGSIZE*NGROUPS);
    }

    //main loop
    for(i = 0; i < gdsteps; i++){
        printf("\nGradient Descent Step %d\n", i);
        
        //burnin loops
        for(j = 0; j < burnin; j++){
            writeData(run_seed_dev, &kernel_seed, sizeof(cl_uint));
            kernel_seed++;
            runKernel(clMCMCKernel, WGSIZE, WGSIZE*NGROUPS);
        }
        
        //counting MCMC loop
        runKernel(clZeroKernel, WGSIZE, WGSIZE*cplgrp);
        for(j = 0; j < nloop; j++){
            writeData(run_seed_dev, &kernel_seed, sizeof(cl_uint));
            kernel_seed++;
            runKernel(clMCMCKernel, WGSIZE, WGSIZE*NGROUPS);
            runKernel(clCountKernel, WGSIZE, WGSIZE*cplgrp);
        }
        runKernel(clUpdateKernel, WGSIZE, WGSIZE*cplgrp);

        writeStatus(L, nB, couplings, bimarg, bicount, seqmem, nseqs);
    }
    
    //Note that loop control is split between nloop and nsteps. This is because
    //on some (many) systems there is a watchdog timer that kills any kernel 
    //that takes too long to finish, thus limiting the maximum nsteps. However,
    //we avoid this by running the same kernel nloop times with smaller nsteps.
    //If you set nsteps too high you will get a CL_OUT_OF_RESOURCES error.

    printf("Done!\n");
    cleanUp();
}

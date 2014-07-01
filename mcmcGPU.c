#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <assert.h>
#include <CL/cl.h>
#include "epsilons.h"
#include "clErrors.h"

#define WGSIZE 256
#define NGROUPS 32

//sequences are padded to 32 bit boundaries
#define SWORDS ((L-1)/4+1)  //number of words needed to store a sequence
#define SBYTES (4*SWORDS)   //number of bytes needed to store a sequence

typedef unsigned char uchar;
typedef unsigned int uint;

char* loadProgSource(const char* cFilename, const char* cPreamble, 
                     size_t* szFinalLength){
    // locals 
    FILE* pFileStream = NULL;
    size_t szSourceLength;

    // open the OpenCL source code file
    pFileStream = fopen(cFilename, "rb");
    if(pFileStream == 0){       
        return NULL;
    }

    size_t szPreambleLength = strlen(cPreamble);

    // get the length of the source code
    fseek(pFileStream, 0, SEEK_END); 
    szSourceLength = ftell(pFileStream);
    fseek(pFileStream, 0, SEEK_SET); 

    // allocate a buffer for the source code string and read it in
    char* cSourceString = (char *)malloc(szSourceLength + szPreambleLength + 1); 
    memcpy(cSourceString, cPreamble, szPreambleLength);
    if(fread(cSourceString+szPreambleLength,szSourceLength,1,pFileStream) != 1){
        fclose(pFileStream);
        free(cSourceString);
        return 0;
    }

    // close the file and return the total length of the 
    // combined (preamble + source) string
    fclose(pFileStream);
    if(szFinalLength != 0){
        *szFinalLength = szSourceLength + szPreambleLength;
    }
    cSourceString[szSourceLength + szPreambleLength] = '\0';

    return cSourceString;
}

cl_context clGPUContext;
cl_command_queue clCommandQue;
cl_device_id *clDevices;
cl_program clProgram;
cl_kernel clMCMCKernel;
cl_kernel clInitseqKernel;
cl_kernel clCountKernel;
cl_kernel clUpdateKernel;
cl_kernel clZeroKernel;
char *kernelSrc; 

void dumpPTX(char *fn){
    cl_uint num_devices;
    cl_int errcode = clGetProgramInfo(clProgram, CL_PROGRAM_NUM_DEVICES, 
                                      sizeof(cl_uint), &num_devices, NULL );
    size_t size[num_devices];

    errcode = clGetProgramInfo(clProgram, CL_PROGRAM_BINARY_SIZES, 
                               sizeof(size_t)*num_devices, &size, NULL );
    gpuErrchk(errcode);
    uchar *binary[num_devices];
    uint i;
    for(i = 0; i < num_devices; i++){
        binary[i] = malloc(sizeof(uchar)*size[i]);
    } 
    errcode = clGetProgramInfo(clProgram, CL_PROGRAM_BINARIES, 
                               sizeof(char*)*num_devices, &binary, NULL );
    gpuErrchk(errcode);
    FILE * fpbin = fopen(fn, "wb" );
    if( fpbin == NULL ) {
        fprintf(stderr, "Cannot create file %s \n", fn);
    }
    else {
        fwrite( binary[0], 1, size[0], fpbin );
        fclose( fpbin );
    }
    for(i = 0; i < num_devices; i++){
        free(binary[i]);
    }
}

void initGPU(){
    cl_int errcode;
    cl_uint numPlatforms;
    errcode = clGetPlatformIDs(0, NULL, &numPlatforms);
    gpuErrchk(errcode);
    size_t infoSize;
    char* info;
    const char* attributeNames[5] = { "Name", "Vendor", "Version", 
                                      "Profile", "Extensions" };
    const cl_platform_info attrTypes[5] = { CL_PLATFORM_NAME, 
                                                 CL_PLATFORM_VENDOR, 
                                                 CL_PLATFORM_VERSION, 
                                                 CL_PLATFORM_PROFILE, 
                                                 CL_PLATFORM_EXTENSIONS };
    const int attributeCount = sizeof(attributeNames) / sizeof(char*);

    // get all the platforms and print them
    cl_platform_id *platforms = malloc(sizeof(cl_platform_id)*numPlatforms);
    errcode = clGetPlatformIDs(numPlatforms, platforms, NULL);
    gpuErrchk(errcode);
    int i,j;
    for (i = 0; i < numPlatforms; i++) {
        printf("\n %d. Platform \n", i+1);
        for (j = 0; j < attributeCount; j++) {
            clGetPlatformInfo(platforms[i], attrTypes[j], 0, NULL, &infoSize);
            info = (char*) malloc(infoSize);
            clGetPlatformInfo(platforms[i], attrTypes[j], infoSize, info, NULL);
            printf("  %d.%d %-11s: %s\n", i+1, j+1, attributeNames[j], info);
            free(info);
        }
        printf("\n");
    }

    // set platform property - we just pick the first one
    cl_context_properties properties[] = {CL_CONTEXT_PLATFORM, 
                                          (int) platforms[0], 
                                          0};
    clGPUContext = clCreateContextFromType(properties, CL_DEVICE_TYPE_GPU, 
                                           NULL, NULL, &errcode);
    gpuErrchk(errcode);

    // get the list of GPU devices associated with context
    size_t dataBytes;
    errcode = clGetContextInfo(clGPUContext, CL_CONTEXT_DEVICES, 0, NULL, 
                               &dataBytes);
    clDevices = malloc(dataBytes);
    errcode |= clGetContextInfo(clGPUContext, CL_CONTEXT_DEVICES, dataBytes, 
                                clDevices, NULL);
    gpuErrchk(errcode);

    cl_uint devices_n = 0;
    clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU,100, clDevices, &devices_n);
    
    printf("Devices:\n");
    for(i=0; i<devices_n; i++){
            char buffer[10240];
            cl_uint buf_uint;
            cl_ulong buf_ulong;
            printf("  -- %d --\n", i);
            clGetDeviceInfo(clDevices[i], CL_DEVICE_NAME, 
                            sizeof(buffer), buffer, NULL);
            printf("  DEVICE_NAME = %s\n", buffer);
            clGetDeviceInfo(clDevices[i], CL_DEVICE_VENDOR, 
                            sizeof(buffer), buffer, NULL);
            printf("  DEVICE_VENDOR = %s\n", buffer);
            clGetDeviceInfo(clDevices[i], CL_DEVICE_VERSION, 
                            sizeof(buffer), buffer, NULL);
            printf("  DEVICE_VERSION = %s\n", buffer);
            clGetDeviceInfo(clDevices[i], CL_DRIVER_VERSION, 
                            sizeof(buffer), buffer, NULL);
            printf("  DRIVER_VERSION = %s\n", buffer);
            clGetDeviceInfo(clDevices[i], CL_DEVICE_MAX_COMPUTE_UNITS, 
                            sizeof(buf_uint), &buf_uint, NULL);
            printf("  DEVICE_MAX_COMPUTE_UNITS = %u\n", (uint)buf_uint);
            clGetDeviceInfo(clDevices[i], CL_DEVICE_MAX_CLOCK_FREQUENCY, 
                            sizeof(buf_uint), &buf_uint, NULL);
            printf("  DEVICE_MAX_CLOCK_FREQUENCY = %u\n", (uint)buf_uint);
            clGetDeviceInfo(clDevices[i], CL_DEVICE_GLOBAL_MEM_SIZE, 
                            sizeof(buf_ulong), &buf_ulong, NULL);
            printf("  DEVICE_GLOBAL_MEM_SIZE = %llu\n", 
                   (unsigned long long)buf_ulong);
        }

    //Create a command-queue
    clCommandQue = clCreateCommandQueue(clGPUContext, clDevices[0],0, &errcode);
    gpuErrchk(errcode);
}

void loadKernel(char *kernelName, uint nB, uint L, uint nloop, uint nsteps){
    cl_int errcode;
    
    //load Kernel
    size_t kernelLength;
    kernelSrc = loadProgSource(kernelName, "", &kernelLength);
    assert(kernelSrc != NULL);

    clProgram = clCreateProgramWithSource(clGPUContext, 1, 
                            (const char **)&kernelSrc, &kernelLength, &errcode);
    gpuErrchk(errcode);

    float gamma = 0.0005;
    uint nseqs = WGSIZE*NGROUPS*nloop;
    
    char options[1000];
    sprintf(options, "-D NGROUPS=%d -D WGSIZE=%d -D SEED=%d "
                     "-D nB=%d -D L=%d -D nsteps=%d "
                     "-D nseqs=%d -D gamma=%g -cl-nv-verbose -Werror", 
                     NGROUPS, WGSIZE, rand(), nB, L, 
                     nsteps, nseqs, gamma);
    printf("Option: %s\n", options);
    errcode = clBuildProgram(clProgram, 0, NULL, options, NULL, NULL);

    //if (errcode == CL_BUILD_PROGRAM_FAILURE) {
        // Determine the size of the log
        size_t log_size;
        clGetProgramBuildInfo(clProgram, clDevices[0], 
                              CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char *log = malloc(log_size);

        // Get the log
        clGetProgramBuildInfo(clProgram, clDevices[0], 
                              CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        printf("%s\n", log);
    //}
    gpuErrchk(errcode);
 

}

void setupMCMCKernel(cl_mem J_dev, cl_mem run_seed_dev, cl_mem savedseq_dev){
    cl_int errcode;

    clMCMCKernel = clCreateKernel(clProgram, "metropolis", &errcode);
    gpuErrchk(errcode);

    errcode |= clSetKernelArg(clMCMCKernel, 0, 
               sizeof(cl_mem), (void *)&J_dev);
    errcode |= clSetKernelArg(clMCMCKernel, 1, 
               sizeof(cl_mem), (void *)&run_seed_dev);
    errcode |= clSetKernelArg(clMCMCKernel, 2, 
               sizeof(cl_mem), (void *)&savedseq_dev);
    gpuErrchk(errcode);
}

void setupUpdateKernel(cl_mem target_bimarg_dev, 
                       cl_mem bicount_dev, 
                       cl_mem J_dev){
    cl_int errcode;

    clUpdateKernel = clCreateKernel(clProgram, "updateCouplings", &errcode);
    gpuErrchk(errcode);

    errcode |= clSetKernelArg(clUpdateKernel, 0, 
               sizeof(cl_mem), (void *)&target_bimarg_dev);
    errcode |= clSetKernelArg(clUpdateKernel, 1, 
               sizeof(cl_mem), (void *)&bicount_dev);
    errcode |= clSetKernelArg(clUpdateKernel, 2, 
               sizeof(cl_mem), (void *)&J_dev);
    gpuErrchk(errcode);
}

void setupCountKernel(cl_mem bicount_dev,  
                      cl_mem savedseq_dev, 
                      cl_mem pairI_dev, 
                      cl_mem pairJ_dev){
    cl_int errcode;
    clCountKernel = clCreateKernel(clProgram, "countSeqs", &errcode);
    gpuErrchk(errcode);

    errcode |= clSetKernelArg(clCountKernel, 0, 
               sizeof(cl_mem), (void *)&bicount_dev);
    errcode |= clSetKernelArg(clCountKernel, 1, 
               sizeof(cl_mem), (void *)&savedseq_dev);
    errcode |= clSetKernelArg(clCountKernel, 2, 
               sizeof(cl_mem), (void *)&pairI_dev);
    errcode |= clSetKernelArg(clCountKernel, 3, 
               sizeof(cl_mem), (void *)&pairJ_dev);
    gpuErrchk(errcode);
}

void setupInitseqKernel(cl_mem startseq_dev, cl_mem savedseq_dev){
    cl_int errcode;
    clInitseqKernel = clCreateKernel(clProgram, "initSeqMem", &errcode);
    gpuErrchk(errcode);

    errcode |= clSetKernelArg(clInitseqKernel, 0, 
               sizeof(cl_mem), (void *)&startseq_dev);
    errcode |= clSetKernelArg(clInitseqKernel, 1, 
               sizeof(cl_mem), (void *)&savedseq_dev);
    gpuErrchk(errcode);
}

void setupZeroKernel(cl_mem bicounts){
    cl_int errcode;
    clZeroKernel = clCreateKernel(clProgram, "zeroBicounts", &errcode);
    gpuErrchk(errcode);

    errcode |= clSetKernelArg(clZeroKernel, 0, 
               sizeof(cl_mem), (void *)&bicounts);
    gpuErrchk(errcode);
}

void initSeqMem(){
    cl_int errcode;
    size_t localWorkSize[1], globalWorkSize[1];
    
    localWorkSize[0] = WGSIZE;
    globalWorkSize[0] = WGSIZE*NGROUPS; 
    errcode = clEnqueueNDRangeKernel(clCommandQue, 
               clInitseqKernel, 1, NULL, globalWorkSize, 
               localWorkSize, 0, NULL, NULL);
    gpuErrchk(errcode);
}

void runMetropolis(){
    cl_int errcode;
    size_t localWorkSize[1], globalWorkSize[1];
    
    localWorkSize[0] = WGSIZE;
    globalWorkSize[0] = WGSIZE*NGROUPS; 
    errcode = clEnqueueNDRangeKernel(clCommandQue, 
               clMCMCKernel, 1, NULL, globalWorkSize, 
               localWorkSize, 0, NULL, NULL);
    gpuErrchk(errcode);
}

void countSeqs(cl_int ncouple){
    cl_int errcode;
    size_t localWorkSize[1], globalWorkSize[1];
    
    //global size is minimal multiple of wgsize greater than ncoupling
    localWorkSize[0] = WGSIZE;
    globalWorkSize[0] = WGSIZE*((ncouple-1)/WGSIZE+1); 
    errcode = clEnqueueNDRangeKernel(clCommandQue, 
               clCountKernel, 1, NULL, globalWorkSize, 
               localWorkSize, 0, NULL, NULL);
    gpuErrchk(errcode);
}

void updateCouplings(cl_int ncouple){
    cl_int errcode;
    size_t localWorkSize[1], globalWorkSize[1];

    localWorkSize[0] = WGSIZE;
    globalWorkSize[0] = WGSIZE*((ncouple-1)/WGSIZE+1); 
    errcode = clEnqueueNDRangeKernel(clCommandQue, 
               clUpdateKernel, 1, NULL, globalWorkSize, 
               localWorkSize, 0, NULL, NULL);
    gpuErrchk(errcode);
}

void zeroBicounts(cl_int ncouple){
    cl_int errcode;
    size_t localWorkSize[1], globalWorkSize[1];

    localWorkSize[0] = WGSIZE;
    globalWorkSize[0] = WGSIZE*((ncouple-1)/WGSIZE+1); 
    errcode = clEnqueueNDRangeKernel(clCommandQue, 
               clZeroKernel, 1, NULL, globalWorkSize, 
               localWorkSize, 0, NULL, NULL);
    gpuErrchk(errcode);
}

void readData(void *buf, cl_mem dev_mem, size_t size){
    cl_int errcode;
    errcode = clEnqueueReadBuffer(clCommandQue, 
               dev_mem, CL_TRUE, 0, size, buf,
               0, NULL, NULL);
    gpuErrchk(errcode);
}
void writeData(cl_mem dev_mem, void *buf, size_t size){
    cl_int errcode;
    errcode = clEnqueueWriteBuffer(clCommandQue, dev_mem, CL_TRUE, 
                                   0, size, buf, 0, NULL, NULL);
    gpuErrchk(errcode);
}

//only works in openCL 1.2
//void zeroWordBuffer(cl_mem buffer, size_t bytes){
//    cl_uint zero = 0;
//    clEnqueueFillBuffer(clCommandQue, buffer, &zero, sizeof(cl_uint), 
//                        0, bytes, 0, NULL, NULL);
//}

cl_mem J_dev;
cl_mem bicount_dev;
cl_mem bimarg_dev;
cl_mem pairI_dev;
cl_mem pairJ_dev;
cl_mem run_seed_dev;
cl_mem savedseq_dev;
cl_mem startseq_dev;
cl_uint *pairI, *pairJ;
void setupMem(uint L, uint nB, cl_float *couplings, 
              cl_float *bimarg, uchar *startseq){
    cl_int errcode;
    uint i,j,n;
    uint n_couplings = L*(L-1)*nB*nB/2;
    
    J_dev = clCreateBuffer(clGPUContext, 
           CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 
           sizeof(cl_float)*n_couplings, couplings, &errcode);
    gpuErrchk(errcode);
    bicount_dev = clCreateBuffer(clGPUContext, 
           CL_MEM_READ_WRITE, 
           sizeof(cl_uint)*n_couplings, NULL, &errcode);
    gpuErrchk(errcode);
    if(bimarg != NULL){
        bimarg_dev = clCreateBuffer(clGPUContext, 
               CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 
               sizeof(float)*n_couplings, bimarg, &errcode);
    }else{
        bimarg_dev = clCreateBuffer(clGPUContext, 
               CL_MEM_READ_WRITE, 
               sizeof(float)*n_couplings, NULL, &errcode);
    }
    gpuErrchk(errcode);
    run_seed_dev = clCreateBuffer(clGPUContext, 
           CL_MEM_READ_WRITE, 
           sizeof(cl_uint), NULL, &errcode);
    gpuErrchk(errcode);
    savedseq_dev = clCreateBuffer(clGPUContext, 
           CL_MEM_READ_WRITE, 
           sizeof(cl_uint)*SWORDS*NGROUPS*WGSIZE, NULL, &errcode);
    gpuErrchk(errcode);
    startseq_dev = clCreateBuffer(clGPUContext, 
           CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 
           sizeof(cl_uint)*SWORDS, startseq, &errcode);
    gpuErrchk(errcode);
    
    //set up the map from pair index to pair values
    pairI = malloc(sizeof(cl_uint)*L*(L-1)/2);
    pairJ = malloc(sizeof(cl_uint)*L*(L-1)/2);
    n = 0;
    for(i = 0; i < L-1; i++){
        for(j = i+1; j < L; j++){
            pairI[n] = i;
            pairJ[n] = j;
            n++;
        }
    }
    pairI_dev = clCreateBuffer(clGPUContext, 
           CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
           sizeof(cl_uint)*L*(L-1)/2, pairI, &errcode);
    pairJ_dev = clCreateBuffer(clGPUContext, 
           CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
           sizeof(cl_uint)*L*(L-1)/2, pairJ, &errcode);
    gpuErrchk(errcode);

}

void freeMem(){
    clReleaseMemObject(J_dev);
    clReleaseMemObject(bicount_dev);
    clReleaseMemObject(pairI_dev);
    clReleaseMemObject(pairJ_dev);
 
    free(clDevices);
    free(kernelSrc);
    clReleaseContext(clGPUContext);
    clReleaseKernel(clMCMCKernel);
    clReleaseKernel(clCountKernel);
    clReleaseKernel(clUpdateKernel);
    clReleaseProgram(clProgram);
    clReleaseCommandQueue(clCommandQue);
}

void writeStatus(uint L, uint nB, float *couplings, float *bimarg, 
                 uint *bicount, uint *seqmem, uint nseqs){
    uint i,j,a;

    uint nPairs = L*(L-1)/2;
    uint n_couplings = nPairs*nB*nB;

    readData(couplings, J_dev, sizeof(uint)*n_couplings);
    readData(bicount, bicount_dev, sizeof(cl_uint)*n_couplings);
    readData(seqmem, savedseq_dev, sizeof(cl_uint)*WGSIZE*NGROUPS*SWORDS);
    
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

    f = fopen("finalseqs", "wb");
    fwrite(seqmem, sizeof(cl_uint), WGSIZE*NGROUPS*SWORDS, f);
    fclose(f);
}

int
main(int argc, char** argv){
    int i,j,l,n,a,b;
    FILE *f;

    if(argc != 9 && argc != 11){
        printf("Usage: ./a.out bimarg gdsteps initialBurnin burnloop nloop "
               "niter alphabet startseq [couplings finalseqs]\n");
        exit(1);
    }

    printf("Initialization\n===============\n");
 
    cl_int errcode;
    // set seed for rand()
    //srand(2007);
    srand((unsigned)time(NULL));
    
    
    ////////////// 
    //allocate all memory, read in args and data
    uint nPairs, nComb; 
    double *bimarg_in = readEM(&nComb, &nPairs, argv[1]);
    uint L = (int)((1+sqrt(1+8*nPairs))/2); //fp error possible?
    uint nB = (int)(sqrt(nComb));
    printf("\nnBases %d  seqLen %d\n", nB, L);

    uint n_couplings = L*(L-1)*nB*nB/2;
    cl_uint *bicount = malloc(sizeof(cl_uint)*n_couplings);
    cl_float *couplings = malloc(sizeof(float)*nComb*nPairs);

    uint gdsteps = atoi(argv[2]);
    uint burnstart = atoi(argv[3]);
    uint burnin = atoi(argv[4]);
    uint nloop = atoi(argv[5]);
    uint nsteps = atoi(argv[6]);
    uint nseqs = WGSIZE*NGROUPS*nloop;

    uint *seqmem = malloc(sizeof(cl_uint)*WGSIZE*NGROUPS*SWORDS);
    uchar *seqs = (uchar*)seqmem;

    char *alphabet = argv[7];
    uint *startseqmem = malloc(sizeof(uint)*SWORDS);
    for(i = 0; i < SWORDS; i++){
        startseqmem[i] = 0;
    }
    uchar *startseq = (uchar*)startseqmem;
    printf("Initial sequence: ");
    for(i = 0; i < L; i++){
        //XXX check bounds
        startseq[i] = (uchar)(strchr(alphabet, argv[7][i]) - alphabet); 
        printf("%d ", startseq[i]);
    }
    printf("\n");

    cl_float *bimarg = malloc(sizeof(cl_float)*n_couplings);
    for(i = 0; i < nPairs*nComb; i++){
        bimarg[i] = bimarg_in[i];
    }
    
    if(argc == 11){
        f = fopen(argv[9], "rb");
        fread(couplings, sizeof(cl_float), n_couplings, f);
        fclose(f);
    }else{
        for(i = 0; i < nComb*nPairs; i++){
            if(bimarg[i] == 0){
                couplings[i] = INFINITY;
            }
            else{
                couplings[i] = 0;
            }
        }
    }

    printf("Initial couplings: ");
    for(a = 0; a < 10; a++){
        printf("%.15g ", couplings[a]);
    }
    printf(" ...\n");

    ////////////// 
    //Set up kernels and device memory & transfer data to GPU
    initGPU();
    loadKernel("metropolis.cl", nB, L, nloop, nsteps);
    //dumpPTX("metropolis.ptx");
 
    setupMem(L, nB, couplings, bimarg, startseq);
    setupMCMCKernel(J_dev, run_seed_dev, savedseq_dev);
    setupUpdateKernel(bimarg_dev, bicount_dev, J_dev);
    setupInitseqKernel(startseq_dev, savedseq_dev);
    setupCountKernel(bicount_dev, savedseq_dev, pairI_dev, pairJ_dev);
    setupZeroKernel(bicount_dev);
    
    if(argc == 9){
        initSeqMem(); //kernel call, fills in seqmem with initial sequence
    } else{
        f = fopen(argv[10], "rb");
        fread(seqmem, sizeof(cl_uint), WGSIZE*NGROUPS*SWORDS, f);
        fclose(f);
        writeData(savedseq_dev, seqmem, sizeof(cl_uint)*WGSIZE*NGROUPS*SWORDS);
    }
    
    ////////////// 
    //perform the computation!
    printf("\n\nMCMC Run\n========\n");
    
    //initial burnin
    uint kernel_seed = 0;
    printf("Initial Burnin for %d steps:\n", burnstart);
    for(j = 0; j < burnstart; j++){
        writeData(run_seed_dev, &kernel_seed, sizeof(cl_uint));
        kernel_seed++;
        runMetropolis();
    }

    //main loop
    for(i = 0; i < gdsteps; i++){
        printf("\nGradient Descent Step %d\n", i);
        
        //burnin loops
        for(j = 0; j < burnin; j++){
            writeData(run_seed_dev, &kernel_seed, sizeof(cl_uint));
            kernel_seed++;
            runMetropolis();
        }
        
        //counting MCMC loop
        zeroBicounts(n_couplings);
        for(j = 0; j < nloop; j++){
            writeData(run_seed_dev, &kernel_seed, sizeof(cl_uint));
            kernel_seed++;
            runMetropolis();
            countSeqs(n_couplings);
        }
        updateCouplings(n_couplings);

        writeStatus(L, nB, couplings, bimarg, bicount, seqmem, nseqs);
    }

    printf("Done!\n");
    freeMem();
}

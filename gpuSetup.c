#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include "clErrors.h"
#include "gpuSetup.h"

//manages the GPU, for one device, one queue

static cl_context clGPUContext;
static cl_command_queue clCommandQue;
static cl_device_id *clDevices;
static cl_program clProgram;

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
    printf("Platform Information:");
    int i,j;
    for (i = 0; i < numPlatforms; i++) {
        printf("\n %d. Platform %d \n", i+1, i+1);
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
                                     (cl_context_properties)(platforms[0]), 0};
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
        size_t buf_sizet;
        printf("  Device %d:\n", i);
        #define disp(name, deref, dst, fmt) \
            clGetDeviceInfo(clDevices[i], name, \
                            sizeof(dst), deref dst, NULL); \
            printf("    " #name " = " fmt "\n", dst);
        disp(CL_DEVICE_NAME, , buffer, "%s");
        disp(CL_DEVICE_VENDOR, , buffer, "%s");
        disp(CL_DEVICE_VERSION, , buffer, "%s");
        disp(CL_DRIVER_VERSION, , buffer, "%s");
        disp(CL_DEVICE_MAX_CLOCK_FREQUENCY, &, buf_uint, "%u");
        disp(CL_DEVICE_MAX_COMPUTE_UNITS, &, buf_uint, "%u");
        disp(CL_DEVICE_MAX_WORK_GROUP_SIZE, &, buf_sizet, "%zu");
        disp(CL_DEVICE_GLOBAL_MEM_SIZE, &, buf_ulong, "%llu");
        disp(CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, &, buf_ulong, "%llu");
        disp(CL_DEVICE_LOCAL_MEM_SIZE, &, buf_ulong, "%llu");
        disp(CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, &, buf_ulong, "%llu");
        #undef disp
    }
    printf("\n");

    //Create a command-queue
    clCommandQue = clCreateCommandQueue(clGPUContext, clDevices[0],0, &errcode);
    gpuErrchk(errcode);
}

char *loadSource(const char *fn, const char *preamble, size_t *len_out){
    // locals 
    FILE* f = NULL;
    size_t source_len;

    // open the OpenCL source code file
    f = fopen(fn, "rb");
    if(f == 0){       
        return NULL;
    }

    size_t preamble_len = strlen(preamble);

    // get the length of the source code
    fseek(f, 0, SEEK_END); 
    source_len = ftell(f);
    fseek(f, 0, SEEK_SET); 

    // allocate a buffer for the source code string and read it in
    char* source = malloc(source_len + preamble_len + 1); 
    memcpy(source, preamble, preamble_len);
    if(fread(source + preamble_len, source_len, 1, f) != 1){
        fclose(f);
        free(source);
        return 0;
    }

    // close the file and return the total length of the 
    // combined (preamble + source) string
    fclose(f);
    if(len_out != 0){
        *len_out = source_len + preamble_len;
    }
    source[source_len + preamble_len] = '\0';

    return source;
}

void dumpPTX(char *fn){
    cl_uint num_devices;
    cl_int errcode = clGetProgramInfo(clProgram, CL_PROGRAM_NUM_DEVICES, 
                                      sizeof(cl_uint), &num_devices, NULL );
    size_t size[num_devices];

    errcode = clGetProgramInfo(clProgram, CL_PROGRAM_BINARY_SIZES, 
                               sizeof(size_t)*num_devices, &size, NULL );
    gpuErrchk(errcode);
    unsigned char *binary[num_devices];
    unsigned int i;
    for(i = 0; i < num_devices; i++){
        binary[i] = malloc(sizeof(unsigned char)*size[i]);
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


void loadKernel(char *kernelName, char *options){
    cl_int errcode;
    
    //load Kernel
    size_t kernelLength;
    char *kernelSrc = loadSource(kernelName, "", &kernelLength);
    assert(kernelSrc != NULL);

    clProgram = clCreateProgramWithSource(clGPUContext, 1, 
                            (const char **)&kernelSrc, &kernelLength, &errcode);
    gpuErrchk(errcode);
    errcode = clBuildProgram(clProgram, 0, NULL, options, NULL, NULL);

    //Print out Build log
    size_t log_size;
    clGetProgramBuildInfo(clProgram, clDevices[0], 
                          CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
    char *log = malloc(log_size);
    clGetProgramBuildInfo(clProgram, clDevices[0], 
                          CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
    printf("Compilation log:%s\n", log);
    gpuErrchk(errcode);
    free(kernelSrc);
}

cl_kernel setupKernel(char *kernelName, unsigned int nargs, ...){
    cl_int errcode;
    unsigned int i;

    cl_kernel kernel = clCreateKernel(clProgram, kernelName, &errcode);
    gpuErrchk(errcode);

    va_list argp;
    va_start(argp, nargs); 
    for(i = 0; i < nargs; i++){ 
        cl_mem arg = va_arg(argp, cl_mem);
        errcode |= clSetKernelArg(kernel, i, 
                   sizeof(cl_mem), (void *)&arg);
    }
    va_end(argp);
    gpuErrchk(errcode);

    return kernel;
}

void runKernel(cl_kernel kernel, unsigned int wgsize, unsigned int gsize){
    cl_int errcode;
    size_t localWorkSize[1], globalWorkSize[1];
    
    localWorkSize[0] = wgsize;
    globalWorkSize[0] = gsize; 
    errcode = clEnqueueNDRangeKernel(clCommandQue, 
               kernel, 1, NULL, globalWorkSize, 
               localWorkSize, 0, NULL, NULL);
    gpuErrchk(errcode);
}

cl_mem createRWBufferEmpty(size_t size){
    cl_int errcode;
    cl_mem dat = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, 
                                size, NULL, &errcode);
    gpuErrchk(errcode);
    return dat;
}

cl_mem createRWBufferFilled(size_t size, void *host_ptr){
    cl_int errcode;
    assert(host_ptr != NULL);
    cl_mem dat = clCreateBuffer(clGPUContext, 
                                CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 
                                size, host_ptr, &errcode);
    gpuErrchk(errcode);
    return dat;
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

void freeGPU(){
    free(clDevices);
    clReleaseContext(clGPUContext);
    clReleaseCommandQueue(clCommandQue);
    clReleaseProgram(clProgram);
}

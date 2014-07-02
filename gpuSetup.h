#ifndef GPUSETUP_H
#define GPUSETUP_H
#include <stdarg.h>
#include <CL/cl.h>

void initGPU();
void freeGPU();
char *loadSource(const char *fn, const char *preamble, size_t *len_out);
void loadKernel(char *kernelName, char *options);
cl_kernel setupKernel(char *kernelName, unsigned int nargs, ...);
void runKernel(cl_kernel kernel, unsigned int wgsize, unsigned int gsize);
cl_mem createRWBufferEmpty(size_t size);
cl_mem createRWBufferFilled(size_t size, void *host_ptr);
void readData(void *buf, cl_mem dev_mem, size_t size);
void writeData(cl_mem dev_mem, void *buf, size_t size);

#endif

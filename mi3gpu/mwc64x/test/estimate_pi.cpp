#include "opencl_device.hpp"
#include "opencl_buffer.hpp"

int main(int argc,char *argv[])
{
	try{
		if(argc<4){
			fprintf(stderr, "Usage: estimate_pi rng_path.cl work_items log2n\n");
			exit(1);
		}
		
		OpenCLDevice ocd;
		ocd.AppendCppDef("-I. -I../cl");
		ocd.AddSourceFile(argv[1]);
		
		unsigned workItems=atoi(argv[2]);
		unsigned log2n=atoi(argv[3]);
		
		cl::CommandQueue q=ocd.CreateCommandQueue();
		cl::KernelFunctor k=ocd.GetKernelFunctor("EstimatePi", q, workItems);
		
		uint64_t n=1ULL<<log2n;
		
		OpenCLBuffer<uint64_t> tmp(ocd.GetContext(), workItems, std::string("accumulators"));
		
		k(n, 0ULL, tmp.buffer()).wait();
		tmp.copyFromDevice(q);
		
		uint64_t total=0;
		for(unsigned i=0;i<workItems;i++)
			total+=tmp[i];
			
		double frac=total/(double)n;
		
		fprintf(stderr, " %20s, work_items=%3u, n=2^%u, hits=%llu, pi=%lf\n", argv[1], workItems, log2n, total, 4*frac);
		
		return !tmp[0];
		
	}catch(cl::Error &e){
		fprintf(stderr, "cl::Error : %s, %s\n", e.what(), clErrorToString(e.err()));
		return 1;
	}catch(std::exception &e){
		fprintf(stderr, "Exception : %s\n", e.what());
		return 1;
	}
		
	return 0;
}

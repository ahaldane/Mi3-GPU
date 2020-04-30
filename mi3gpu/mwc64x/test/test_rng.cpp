#include "opencl_device.hpp"
#include "opencl_buffer.hpp"

int main(int argc,char *argv[])
{
	try{
		if(argc<2){
			fprintf(stderr, "Usage: test_rng rng_path.cl\n");
			exit(1);
		}
		
		OpenCLDevice ocd;
		ocd.AppendCppDef("-I. -I../cl");
		ocd.AddSourceFile(argv[1]);
		
		cl::CommandQueue q=ocd.CreateCommandQueue();
		cl::KernelFunctor k=ocd.GetKernelFunctor("TestRng", q);
		
		OpenCLBuffer<uint32_t> tmp(ocd.GetContext(), 1u, std::string("success"));
		
		k(tmp.buffer()).wait();
		tmp.copyFromDevice(q);
		
		fprintf(stderr, " %s, %s\n", argv[1], tmp[0] ? "ok": "FAIL");
		
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

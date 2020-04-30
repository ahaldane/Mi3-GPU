#include "opencl_device.hpp"
#include "opencl_buffer.hpp"

int main(int argc,char *argv[])
{
	try{
		if(argc<2){
			fprintf(stderr, "Usage: dump_samples rng_path.cl [work_items] [total_samples]\n");
			exit(1);
		}
		
		unsigned workItems=4;
		if(argc>2){
			workItems=atoi(argv[2]);
		}
		
		cl_uint totalSamples=1U<<16;
		if(argc>3){
			totalSamples=atoi(argv[3]);
		}
		
		OpenCLDevice ocd;
		ocd.AppendCppDef("-I. -I../cl");
		ocd.AddSourceFile(argv[1]);
		
		cl::CommandQueue q=ocd.CreateCommandQueue();
		cl::KernelFunctor k=ocd.GetKernelFunctor("DumpSamples", q, workItems);
		
		OpenCLBuffer<uint32_t> output(ocd.GetContext(), totalSamples, std::string("output"));
		
		k(totalSamples, output.buffer()).wait();
		output.copyFromDevice(q);
		
		for(unsigned i=0;i<totalSamples;i++){
			fprintf(stdout, "%08x\n", output[i]);
		}
		
	}catch(cl::Error &e){
		fprintf(stderr, "cl::Error : %s, %s\n", e.what(), clErrorToString(e.err()));
		return 1;
	}catch(std::exception &e){
		fprintf(stderr, "Exception : %s\n", e.what());
		return 1;
	}catch(...){
		fprintf(stderr, "Caught Unknown Exception\n");
		return 1;
	}
		
	return 0;
}

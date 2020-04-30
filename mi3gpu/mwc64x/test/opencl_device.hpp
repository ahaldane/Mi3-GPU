/*
Stupid OpenCL wrapper By David Thomas, dt10@imperial.ac.uk
This is provided under BSD, full license is with whatever package
of mine it came from.
See http://www.doc.ic.ac.uk/~dt10/research
*/
#ifndef opencl_wrapper_hpp
#define opencl_wrapper_hpp

#define __CL_ENABLE_EXCEPTIONS
#include "CL/cl.hpp"

#include <list>
#include <stdexcept>
#include <fstream>
#include <cstdio>
#include <iostream>
#include <cassert>

template<class T>
bool operator!(const cl::detail::Wrapper<T> &x)
{
	return NULL == x();
}

const char *clErrorToString(cl_int err)
{
    switch (err) {
        case CL_SUCCESS:                            return "Success!";
        case CL_DEVICE_NOT_FOUND:                   return "Device not found.";
        case CL_DEVICE_NOT_AVAILABLE:               return "Device not available";
        case CL_COMPILER_NOT_AVAILABLE:             return "Compiler not available";
        case CL_MEM_OBJECT_ALLOCATION_FAILURE:      return "Memory object allocation failure";
        case CL_OUT_OF_RESOURCES:                   return "Out of resources";
        case CL_OUT_OF_HOST_MEMORY:                 return "Out of host memory";
        case CL_PROFILING_INFO_NOT_AVAILABLE:       return "Profiling information not available";
        case CL_MEM_COPY_OVERLAP:                   return "Memory copy overlap";
        case CL_IMAGE_FORMAT_MISMATCH:              return "Image format mismatch";
        case CL_IMAGE_FORMAT_NOT_SUPPORTED:         return "Image format not supported";
        case CL_BUILD_PROGRAM_FAILURE:              return "Program build failure";
        case CL_MAP_FAILURE:                        return "Map failure";
        case CL_INVALID_VALUE:                      return "Invalid value";
        case CL_INVALID_DEVICE_TYPE:                return "Invalid device type";
        case CL_INVALID_PLATFORM:                   return "Invalid platform";
        case CL_INVALID_DEVICE:                     return "Invalid device";
        case CL_INVALID_CONTEXT:                    return "Invalid context";
        case CL_INVALID_QUEUE_PROPERTIES:           return "Invalid queue properties";
        case CL_INVALID_COMMAND_QUEUE:              return "Invalid command queue";
        case CL_INVALID_HOST_PTR:                   return "Invalid host pointer";
        case CL_INVALID_MEM_OBJECT:                 return "Invalid memory object";
        case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:    return "Invalid image format descriptor";
        case CL_INVALID_IMAGE_SIZE:                 return "Invalid image size";
        case CL_INVALID_SAMPLER:                    return "Invalid sampler";
        case CL_INVALID_BINARY:                     return "Invalid binary";
        case CL_INVALID_BUILD_OPTIONS:              return "Invalid build options";
        case CL_INVALID_PROGRAM:                    return "Invalid program";
        case CL_INVALID_PROGRAM_EXECUTABLE:         return "Invalid program executable";
        case CL_INVALID_KERNEL_NAME:                return "Invalid kernel name";
        case CL_INVALID_KERNEL_DEFINITION:          return "Invalid kernel definition";
        case CL_INVALID_KERNEL:                     return "Invalid kernel";
        case CL_INVALID_ARG_INDEX:                  return "Invalid argument index";
        case CL_INVALID_ARG_VALUE:                  return "Invalid argument value";
        case CL_INVALID_ARG_SIZE:                   return "Invalid argument size";
        case CL_INVALID_KERNEL_ARGS:                return "Invalid kernel arguments";
        case CL_INVALID_WORK_DIMENSION:             return "Invalid work dimension";
        case CL_INVALID_WORK_GROUP_SIZE:            return "Invalid work group size";
        case CL_INVALID_WORK_ITEM_SIZE:             return "Invalid work item size";
        case CL_INVALID_GLOBAL_OFFSET:              return "Invalid global offset";
        case CL_INVALID_EVENT_WAIT_LIST:            return "Invalid event wait list";
        case CL_INVALID_EVENT:                      return "Invalid event";
        case CL_INVALID_OPERATION:                  return "Invalid operation";
        case CL_INVALID_GL_OBJECT:                  return "Invalid OpenGL object";
        case CL_INVALID_BUFFER_SIZE:                return "Invalid buffer size";
        case CL_INVALID_MIP_LEVEL:                  return "Invalid mip-map level";
        default: return "<Unknown error code>";
    }
}


class OpenCLDevice
{
private:
	


	int m_platformSelector, m_deviceSelector;

	std::list<std::string> m_strings;
	std::string m_cppDefs;
	cl::Program::Sources	m_sources;
	cl::Program m_program;

	cl::Platform m_platform;
	cl::Context m_context;
	cl::Device m_device;

	void ChoosePlatform()
	{
		m_platform=cl::Platform();
		m_context=cl::Context();
		m_device=cl::Device();
		m_program=cl::Program();
		
		std::vector< cl::Platform > platformList;
        cl::Platform::get(&platformList);
		if(platformList.size()==0)
			throw std::runtime_error("OpenCLWrapper::ChoosePlatform - Didn't get any platforms from cl::Platform::get.");
		if(m_platformSelector>=platformList.size())
			throw std::runtime_error("OpenCLWrapper::ChoosePlatform - Platform selector index is too large.");
		
		m_platform=platformList[m_platformSelector];
	}
	
	void ChooseDevice()
	{
		m_device=cl::Device();
		m_program=cl::Program();
		
		std::vector<cl::Device> devices;
        devices = GetContext().getInfo<CL_CONTEXT_DEVICES>();
		
		if(devices.size()==0)
			throw std::runtime_error("OpenCLWrapper::ChooseDevice - Didn't get any platforms from cl::Platform::getInfo.");
		if(m_deviceSelector>=devices.size())
			throw std::runtime_error("OpenCLWrapper::ChooseDevice - Device selector index is too large.");
		
		m_device=devices[m_deviceSelector];
	}
	
	OpenCLDevice(OpenCLDevice &);
	OpenCLDevice &operator=(OpenCLDevice &);
public:
	OpenCLDevice()
		: m_platformSelector(0)
		, m_deviceSelector(0)
	{}
		
	void ListDevices(std::ostream &dst)
	{
		std::vector< cl::Platform > platforms;
        cl::Platform::get(&platforms);
		dst<<"Found "<<platforms.size()<<" platforms\n";
		for(unsigned i=0;i<platforms.size();i++){
			cl::Platform p=platforms[i];
			dst<<"Platform "<<i<<": Name=\""<<p.getInfo<CL_PLATFORM_NAME>()<<"\"; Vendor=\""<<p.getInfo<CL_PLATFORM_VENDOR>()<<"\"\n";
			cl_context_properties cprops[3] = {CL_CONTEXT_PLATFORM, (cl_context_properties)(p)(), 0};
			cl::Context ctxt(CL_DEVICE_TYPE_ALL, cprops, NULL, NULL);
			
			std::vector<cl::Device> devices;
			devices = GetContext().getInfo<CL_CONTEXT_DEVICES>();
			dst<<"  Found "<<devices.size()<<" devices\n";
			for(unsigned j=0;j<devices.size();j++){
				cl::Device device=devices[j];
				dst<<"    Device "<<i<<"."<<j<<": Name=\""<<device.getInfo<CL_DEVICE_NAME>()<<"\"\n";
				dst<<"      NumComputeUnits="<<device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
				dst<<", ClockRate="<<device.getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>()/1000.0<<" GHz\n";
			}
		}
	}

	void AddSourceFile(std::string path)
	{
		std::ifstream file(path.c_str());
		if(!file.is_open())
			throw std::runtime_error("OpenCLWrapper::AddSourceFile - Couldn't open kernel source file '"+path+"'");
		std::string code(std::istreambuf_iterator<char>(file), (std::istreambuf_iterator<char>()));
		AddSourceCode(code);
	}
	
	void AddSourceCode(std::string text)
	{
		m_program=cl::Program();
		m_strings.push_back(text);
		m_sources.push_back(std::make_pair(m_strings.back().c_str(), m_strings.back().length()+1));
	}
	
	void ClearSource()
	{
		m_program=cl::Program();
		m_sources.clear();
		m_strings.clear();
	}
	
	void AppendCppDef(std::string s)
	{
		m_cppDefs=m_cppDefs+" "+s;
	}
	
	void Compile()
	{
		if(!m_program){
			assert(!!GetContext());
			cl_int e;
			cl::Program prog(GetContext(), m_sources);
			try{
				std::vector<cl::Device> devices(1, GetDevice());
				prog.build(devices,m_cppDefs.c_str());
				m_program=prog;
			}catch(...){
				std::cerr<<prog.getBuildInfo<CL_PROGRAM_BUILD_LOG>(m_device)<<"\n";
				throw;
			}
		}
	}
	
	cl::Platform GetPlatform()
	{
		if(!m_platform)
			ChoosePlatform();
		return m_platform;
	}
	
	cl::Context GetContext()
	{
		if(!m_context){
			cl_context_properties cprops[3] = {CL_CONTEXT_PLATFORM, (cl_context_properties)(GetPlatform())(), 0};
			
			m_context=cl::Context(CL_DEVICE_TYPE_ALL, cprops, NULL, NULL);
			m_device=cl::Device();
			m_program=cl::Program();
		}
		return m_context;
	}
	
	cl::Device GetDevice()
	{
		if(!m_device){
			ChooseDevice();
		}
		return m_device;
	}
	
	cl::Program GetProgram()
	{
		if(!m_program){
			Compile();
		}
		return m_program;
	}

	cl::Kernel GetKernel(std::string name)
	{ return cl::Kernel(GetProgram(), name.c_str()); }
	
	cl::CommandQueue CreateCommandQueue()
	{ return cl::CommandQueue(GetContext(), GetDevice(), 0); }
	
	cl::KernelFunctor GetKernelFunctor(std::string name, cl::CommandQueue queue, const cl::NDRange& global, const cl::NDRange& local)
	{ return GetKernel(name).bind(queue, global, local); }
	
	cl::KernelFunctor GetKernelFunctor(std::string name, cl::CommandQueue queue, const cl::NDRange& global)
	{ return GetKernelFunctor(name, queue, global, cl::NullRange); }
	
	cl::KernelFunctor GetKernelFunctor(std::string name, cl::CommandQueue queue)
	{ return GetKernelFunctor(name, queue, cl::NDRange(1), cl::NullRange); }
};

#endif

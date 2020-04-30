/*
Stupid OpenCL wrapper By David Thomas, dt10@imperial.ac.uk
This is provided under BSD, full license is with whatever package
of mine it came from.
See http://www.doc.ic.ac.uk/~dt10/research
*/
#ifndef opencl_buffer_hpp
#define opencl_buffer_hpp

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <iterator>
#include <vector>
#include <utility>
#include <sstream>

#define __CL_ENABLE_EXCEPTIONS
#include "CL/cl.hpp"


/*! This is copy-constructible, but note that it acts as a reference-counted
	buffer, so after the copy both things point at the same memory and buffer
	(just like cl::Buffer and boost::smart_ptr)
*/
template<class T>
class OpenCLBuffer
{
private:
	// This is a dumb version of boost::shared_array, as people moan about
	// installing boost. Roll on C++0x ...
	struct shared_array{
		unsigned count;
		char *stg;
		
		shared_array(unsigned nb, std::string name)
			: count(1)
			, stg((char*)malloc(nb))
		{
			if(!stg){
				std::stringstream acc;
				acc<<"Out of memory allocating "<<nb<<" bytes for "<<name;
				throw cl::Error(CL_OUT_OF_HOST_MEMORY, acc.str().c_str());
			}
		}
		
		shared_array *AddRef()
		{ count++; }
		
		void Release()
		{
			count--;
			if(count==0){
				free(stg);
				stg=0;
				delete this;
			}
		}
	};
	
	shared_array *m_pStg;
	T *m_pElts;
	size_t m_n, m_cb;
	std::string m_name;

	cl::Buffer m_buffer;

public:
	OpenCLBuffer()
		: m_pStg(0)
		, m_pElts(0)
		, m_n(0)
		, m_cb(0)
		, m_name("<empty>")
	{}
		
	OpenCLBuffer(const OpenCLBuffer &src)
		: m_pStg(src.m_pStg ? src.m_pStg->AddRef() : 0)
		, m_pElts(src.m_pElts)
		, m_n(src.m_n)
		, m_cb(src.m_cb)
		, m_name(src.m_name)
	{}
		
	OpenCLBuffer &operator=(const OpenCLBuffer &src)
	{
		if(this!=&src){
			release();
			m_pStg = (src.m_pStg ? src.m_pStg->AddRef() : 0);
			m_pElts=src.m_pElts;
			m_n=src.m_n;
			m_cb=src.m_cb;
			m_name=src.m_name;
		}
		return *this;
	}
		
	OpenCLBuffer(cl::Context ctxt, unsigned n, std::string name="<unnamed>")
		: m_pStg(0)
		, m_pElts(0)
		, m_n(0)
		, m_cb(0)
		, m_name("<empty>")
	{
		create(ctxt, n, name);
	}
	
	void create(cl::Context ctxt, unsigned n, std::string name="<unnamed>")
	{
		release();
		
		try{
			m_cb=sizeof(T)*n;
			unsigned bsize=std::max(sizeof(T), (size_t)32);
			m_pStg=new shared_array(m_cb+bsize, name);
			m_pElts=(T*)(((size_t)(m_pStg->stg)+bsize) - (((size_t)(m_pStg->stg)+bsize)%bsize));
			memset(m_pElts, 0, m_cb);
			m_n=n;
					
			m_buffer=cl::Buffer(ctxt, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sizeof(T)*n, m_pElts);
		}catch(...){
			fprintf(stderr, "Error while creating OpenCLBuffer '%s' with %d elements.\n", name.c_str(), n);
			release();
			throw;
		}
	}
	
  void create(cl::Context ctxt, const std::vector<T> &src, std::string name="<unnamed>")
	{
		create(ctxt, src.size(), name);
		std::copy(src.begin(), src.end(), m_pElts);
	}
	
	void release()
	{
		// Note that if two OpenCLBuffers point to the same thing, then this
		// only disconnects one side - the other one should be fine as both
		// cl::Buffer and boost::shared_array are reference counted.
		m_buffer=cl::Buffer();
		m_pElts=NULL;
		if(m_pStg)
			m_pStg->Release();
		m_pStg=0;
		m_n=0;
		m_cb=0;
		m_name="<empty>";
	}
	
	const T *get() const
	{ return m_pElts; }
	
	T * get()
	{ return m_pElts; }
	
	const T *begin() const
	{ return m_pElts; }
	
	const T *end() const
	{ return m_pElts+m_n; }
	
	T *begin()
	{ return m_pElts; }
	
	T *end()
	{ return m_pElts+m_n; }
	
	
	const T &operator[](unsigned x) const
	{
		if(x>=m_n)
			throw cl::Error(CL_OUT_OF_HOST_MEMORY, ("Out of range indexing reading from '"+m_name+"'").c_str());
		return m_pElts[x];
	}
	
	T &operator[](unsigned x)
	{
		if(x>=m_n)
			throw cl::Error(CL_OUT_OF_HOST_MEMORY, ("Out of range indexing accessing element in '"+m_name+"'").c_str());
		return m_pElts[x];
	}
	
	cl_mem buffer() const
	{ return m_buffer(); }
	
	unsigned count() const
	{ return m_n; }
	
	unsigned size() const
	{ return m_n; }
	
	void copyToDevice(cl::CommandQueue &queue)
	{
		if(m_pElts==NULL)
			throw cl::Error(CL_INVALID_MEM_OBJECT, "copyToDevice - Buffer is not initialised.");
		
		queue.enqueueWriteBuffer(m_buffer, CL_TRUE, 0, m_cb, m_pElts);
	}
	
	cl::Event copyToDeviceAsync(cl::CommandQueue &queue, const cl::Event &prior)
	{
		if(m_pElts==NULL)
			throw cl::Error(CL_INVALID_MEM_OBJECT, "copyToDevice - Buffer is not initialised.");
		
		cl::Event complete;
		std::vector<cl::Event> srcs;
		srcs.push_back(prior);
		queue.enqueueWriteBuffer(m_buffer, CL_FALSE, 0, m_cb, m_pElts, srcs, &complete);
		return complete;
	}
	
	cl::Event copyToDeviceAsync(cl::CommandQueue &queue)
	{
		if(m_pElts==NULL)
			throw cl::Error(CL_INVALID_MEM_OBJECT, "copyToDevice - Buffer is not initialised.");
		
		cl::Event complete;
		queue.enqueueWriteBuffer(m_buffer, CL_FALSE, 0, m_cb, m_pElts, NULL, &complete);
		return complete;
	}
	
	void copyFromDevice(cl::CommandQueue &queue)
	{
		if(m_pElts==NULL)
			throw cl::Error(CL_INVALID_MEM_OBJECT, "copyFromDevice - Buffer is not initialised.");
		
		queue.enqueueReadBuffer(m_buffer, CL_TRUE, 0, m_cb, m_pElts);
	}
	
	cl::Event copyFromDeviceAsync(cl::CommandQueue &queue, cl::Event &prior)
	{
		if(m_pElts==NULL)
			throw cl::Error(CL_INVALID_MEM_OBJECT, "copyFromDevice - Buffer is not initialised.");
		
		cl::Event complete;
		
		std::vector<cl::Event> srcs;
		srcs.push_back(prior);
		queue.enqueueReadBuffer(m_buffer, CL_FALSE, 0, m_cb, m_pElts, &srcs, &complete);
		return complete;
	}
	
	cl::Event copyFromDeviceAsync(cl::CommandQueue &queue)
	{
		if(m_pElts==NULL)
			throw cl::Error(CL_INVALID_MEM_OBJECT, "copyFromDevice - Buffer is not initialised.");
		
		cl::Event complete;
		
		queue.enqueueReadBuffer(m_buffer, CL_FALSE, 0, m_cb, m_pElts, NULL, &complete);
		return complete;
	}
	
	void randomiseHost()
	{
	  FILE * f=fopen("/dev/urandom", "r");
		if(f){
			if(m_n!=fread(m_pElts, sizeof(T), m_n, f))
				throw cl::Error(CL_INVALID_VALUE, "OpenCLBuffer::randomise - Couldn't read from /dev/random.");
			fclose(f); // Leak on exception, but never mind
		}else{
			for(unsigned i=0;i<m_cb;i++){
				((unsigned char*)m_pElts)[i]=rand()&0xFF;
			}
		}
	}
	
	class OutProxy
	{
	private:
		cl::CommandQueue &m_queue;
		OpenCLBuffer &m_buffer;
		cl::Event &m_event;
	public:
		OutProxy(cl::CommandQueue &queue, OpenCLBuffer &buffer, cl::Event &event)
			: m_queue(queue)
			, m_buffer(buffer)
			, m_event(event)
		{}
			
		~OutProxy()
		{
			/* This is tricky, as if we try to do the copy again, and it fails, then we might throw
				while handling an exception. The best case seems to be to surpress any queued
				copies if we are unwinding due to an exception.
			*/
			if(std::uncaught_exception()){
				std::cerr<<"\nWARNING: Pending copyFromDevice surpressed, due to outstanding exception.\n";
			}else{
				m_event.wait();
				m_buffer.copyFromDevice(m_queue);
			}
		}
			
		operator cl::Buffer()
		{ return m_buffer.buffer(); }
	};
	
	cl_mem inCopy(cl::CommandQueue &queue)
	{
		copyToDevice();
		return m_buffer;
	}
	
	OutProxy outCopy(cl::CommandQueue &queue, cl::Event &event)
	{
		if(m_pElts==NULL)
			throw throw cl::Error(CL_INVALID_MEM_OBJECT, "Buffer is not initialised.");
		return OutProxy(queue, *this, event);
	}
	
	OutProxy inOutCopy(cl::CommandQueue &queue, cl::Event &event)
	{
		copyToDevice();
		return OutProxy(queue, *this, event);
	}
};

#endif

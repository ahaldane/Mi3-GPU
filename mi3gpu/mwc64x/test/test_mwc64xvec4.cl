#include "mwc64x.cl"

__kernel void TestRng(__global uint *success)
{
	mwc64xvec4_state_t a={(uint4)(0,0,0,0),(uint4)(1,2,3,4)}, b;
	mwc64x_state_t a32={0, 1}, b32;
	uint lSuccess=1;
	
	for(uint dist=0;(dist<4096 && lSuccess);dist++){
		b=a;
		MWC64XVEC4_Skip(&b, dist);
		MWC64X_Skip(&b32, dist);
		for(uint i=0;i<dist;i++){
			MWC64XVEC4_NextUint4(&a);
			MWC64X_NextUint(&a32);
		}
		lSuccess &= ((a.x.s0==a32.x) && (a.c.s0==a32.c));
		lSuccess &= (all(a.x==b.x) && all(a.c==b.c));
		
	}
	
	*success=lSuccess;
}

// total is the number of words to output, must be divisible by 16*get_global_dim(0)
__kernel void DumpSamples(uint total, __global uint *data)
{
	ulong perStream=total/(4*get_global_size(0));
	
	mwc64xvec4_state_t rng;
	MWC64XVEC4_SeedStreams(&rng, 0, perStream);
	
	for(uint i=0;i<perStream;i++){
		uint4 x=MWC64XVEC4_NextUint4(&rng);
		data[get_global_id(0)*4*perStream+0*perStream+i]=x.s0;
		data[get_global_id(0)*4*perStream+1*perStream+i]=x.s1;
		data[get_global_id(0)*4*perStream+2*perStream+i]=x.s2;
		data[get_global_id(0)*4*perStream+3*perStream+i]=x.s3;
	}
}

__kernel void EstimatePi(ulong n, ulong baseOffset, __global ulong *acc)
{
	mwc64xvec4_state_t rng;
    ulong samplesPerStream=n/(4*get_global_size(0));
    MWC64XVEC4_SeedStreams(&rng, baseOffset, 2*samplesPerStream);
    uint count=0;
    for(uint i=0;i<samplesPerStream;i++){
        ulong4 x=convert_ulong4(MWC64XVEC4_NextUint4(&rng));
        ulong4 y=convert_ulong4(MWC64XVEC4_NextUint4(&rng));
        ulong4 x2=x*x;
        ulong4 y2=y*y;
		long4 hit = -(x2+y2 >= x2);
		count += hit.x+hit.y+hit.z+hit.w;
    }
	acc[get_global_id(0)] = count;
}
